"""Original Matrix adapters exchange encrypted traffic through a local homeserver."""

import sys
import unittest

from standalone.tests import test_platform_sdks as platform_fixture


_MATRIX = r'''
import asyncio, contextlib, io, logging, sqlite3, ssl, time, traceback, urllib.parse
from collections import defaultdict
from aiohttp import web
from gateway.config import PlatformConfig
from plugins.platforms.matrix import adapter as matrix
from hermes_constants import set_hermes_home_override,reset_hermes_home_override,get_hermes_home
from agent.secret_scope import set_secret_scope,reset_secret_scope
room="!fixture:example.invalid"
identities={"fixture-bot":("@bot:example.invalid","BOT"),"fixture-peer":("@peer:example.invalid","PEER")}
report={}
fixture_logs=io.StringIO()
logging.basicConfig(level=logging.WARNING,stream=fixture_logs,force=True)
class Homeserver:
    def __init__(self):
        self.keys=defaultdict(dict)
        self.one_time=defaultdict(lambda:defaultdict(dict))
        self.cross=defaultdict(dict)
        self.account_data={}
        self.pending=defaultdict(list)
        self.events=[]
        self.records=[]
        self.media={}
        self.sequence=0
        self.changed=asyncio.Condition()

    async def bump(self):
        self.sequence+=1
        async with self.changed:self.changed.notify_all()

    def state(self):
        return [{"type":"m.room.encryption","state_key":"","sender":"@bot:example.invalid",
                 "event_id":"$encryption","origin_server_ts":1,
                 "content":{"algorithm":"m.megolm.v1.aes-sha2","rotation_period_msgs":2}},
                *[{"type":"m.room.member","state_key":user,"sender":user,"event_id":"$member-"+device,
                   "origin_server_ts":1,"content":{"membership":"join","displayname":device}}
                  for user,device in identities.values()]]

    async def handle(self,request):
        token=request.headers.get("Authorization","").removeprefix("Bearer ")
        if token not in identities:
            return web.json_response({"errcode":"M_UNKNOWN_TOKEN","error":"Fixture token required"},status=401)
        user,device=identities[token]
        path=request.path
        raw=await request.read()
        body=json.loads(raw) if raw and "application/json" in request.headers.get("Content-Type","") else {}
        self.records.append({"method":request.method,"path":path,"user":user,"body":body,"raw":raw})
        if path=="/_matrix/client/versions":
            result={"versions":["r0.6.1","v1.11"],"unstable_features":{}}
        elif path.endswith("/account/whoami"):
            result={"user_id":user,"device_id":device}
        elif path.endswith("/keys/upload"):
            if body.get("device_keys"):self.keys[user][device]=body["device_keys"]
            self.one_time[user][device].update(body.get("one_time_keys",{}))
            result={"one_time_key_counts":{"signed_curve25519":len(self.one_time[user][device])}}
        elif path.endswith("/keys/query"):
            requested=body.get("device_keys",{})
            result={"device_keys":{target:{key:value for key,value in self.keys[target].items()
                        if not devices or key in devices} for target,devices in requested.items()},"failures":{}}
            for name in ("master_keys","self_signing_keys","user_signing_keys"):
                result[name]={target:self.cross[target][name.removesuffix("s")] for target in requested
                              if name.removesuffix("s") in self.cross[target]}
        elif path.endswith("/keys/claim"):
            result={"one_time_keys":{},"failures":{}}
            for target,devices in body["one_time_keys"].items():
                result["one_time_keys"][target]={}
                for target_device in devices:
                    keys=self.one_time[target][target_device]
                    if keys:
                        key=next(iter(keys));result["one_time_keys"][target][target_device]={key:keys.pop(key)}
        elif path.endswith("/keys/device_signing/upload"):
            self.cross[user].update(body)
            result={}
        elif path.endswith("/keys/signatures/upload"):
            for target,keys in body.items():
                for identifier,value in keys.items():
                    if identifier in self.keys[target]:
                        saved=self.keys[target][identifier].setdefault("signatures",{})
                        for signer,signatures in value.get("signatures",{}).items():
                            saved.setdefault(signer,{}).update(signatures)
            result={"failures":{}}
        elif "/account_data/" in path:
            event_type=path.split("/account_data/",1)[1]
            if request.method=="PUT":
                self.account_data[user,event_type]=body
                result={}
            elif event_type=="m.direct":
                result={other:[room] for other,_ in identities.values() if other!=user}
            elif (user,event_type) in self.account_data:
                result=self.account_data[user,event_type]
            else:
                return web.json_response({"errcode":"M_NOT_FOUND","error":"No account data"},status=404)
        elif path.endswith("/sync"):
            since=request.query.get("since")
            last=self.sequence if since is None else int(since)
            if since is not None and last>=self.sequence and not self.pending[user]:
                async with self.changed:
                    try:await asyncio.wait_for(self.changed.wait(),1)
                    except asyncio.TimeoutError:pass
            pending,self.pending[user]=self.pending[user],[]
            result={"next_batch":str(self.sequence),"to_device":{"events":pending},
                "rooms":{"join":{room:{"state":{"events":self.state() if since is None else []},
                    "timeline":{"events":[event for number,event in self.events if number>last],"limited":False}}}},
                "device_one_time_keys_count":{"signed_curve25519":50},"device_lists":{"changed":[],"left":[]}}
        elif "/sendToDevice/" in path:
            event_type=path.split("/sendToDevice/",1)[1].split("/",1)[0]
            for target,devices in body["messages"].items():
                for content in devices.values():
                    self.pending[target].append({"type":event_type,"sender":user,"content":content})
            await self.bump()
            result={}
        elif "/rooms/" in path and "/send/" in path:
            event_type=path.split("/send/",1)[1].split("/",1)[0]
            event={"type":event_type,"sender":user,"room_id":room,"event_id":"$event-"+str(self.sequence+1),
                   "origin_server_ts":int(time.time()*1000),"content":body}
            # Publish before advancing the visible sync token: a woken long
            # poll must never skip an event already covered by next_batch.
            self.events.append((self.sequence+1,event))
            await self.bump()
            result={"event_id":event["event_id"]}
        elif path.endswith("/joined_members"):
            result={"joined":{target:{"display_name":target_device} for target,target_device in identities.values()}}
        elif "/state/m.room.encryption" in path:
            result=self.state()[0]["content"]
        elif "/state/m.room.member/" in path:
            result={"membership":"join","displayname":"Fixture member"}
        elif "/state/m.room.name" in path:
            result={"name":"Fixture encrypted DM"}
        elif "/state/m.room.create" in path:
            result={"creator":"@bot:example.invalid","room_version":"10"}
        elif path.endswith("/state"):
            result=self.state()
        elif "/upload" in path and "/media/" in path:
            identifier="media-"+str(len(self.media)+1)
            self.media[identifier]=raw
            result={"content_uri":"mxc://example.invalid/"+identifier}
        elif "/download/example.invalid/" in path:
            identifier=path.rsplit("/",1)[-1]
            return web.Response(body=self.media[identifier],content_type="application/octet-stream")
        elif "/rooms/" in path and "/redact/" in path:
            result={"event_id":"$redaction-"+str(len(self.records))}
        elif "/typing/" in path or "/read_markers" in path or "/receipt/" in path:
            result={}
        else:
            return web.json_response({"errcode":"M_NOT_FOUND","error":"Unknown fixture path: "+path},status=404)
        return web.json_response(result)

@contextlib.contextmanager
def profile(name,recovery=None):
    home=scratch/"home"/name
    home.mkdir(parents=True,exist_ok=True)
    home_token=set_hermes_home_override(home)
    secrets={"MATRIX_RECOVERY_KEY_OUTPUT_FILE":str(home/"recovery.txt")}
    if recovery is not None:secrets["MATRIX_RECOVERY_KEY"]=recovery
    secret_token=set_secret_scope(secrets,profile_home=str(home))
    try:yield home
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)

async def until(predicate):
    deadline=asyncio.get_running_loop().time()+15
    while asyncio.get_running_loop().time()<deadline:
        if await predicate():return
        future=asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_later(0.02,future.set_result,None)
        await future
    raise AssertionError("Matrix fixture condition timed out")

async def exercise():
    server=Homeserver()
    app=web.Application()
    app.router.add_route("*","/{tail:.*}",server.handle)
    runner=web.AppRunner(app)
    await runner.setup()
    tls=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(scratch/"cert.pem",scratch/"key.pem")
    site=web.TCPSite(runner,"127.0.0.1",0,ssl_context=tls)
    await site.start()
    origin="https://127.0.0.1:"+str(runner.addresses[0][1])
    adapters={}
    received={"bot":[],"peer":[]}
    report["received"]={}
    async def connect(name,recovery=None):
        with profile(name,recovery):
            adapter=matrix.MatrixAdapter(PlatformConfig(enabled=True,token="fixture-"+name,
                extra={"homeserver":origin,"e2ee_mode":"required","require_mention":False,"user_id":"@"+name+":example.invalid"}))
            async def inbound(event):
                assert get_hermes_home()==scratch/"home"/name
                received[name].append(event)
            adapter.set_message_handler(inbound)
            assert matrix._check_e2ee_deps()
            logging.getLogger("mau.crypto").setLevel(logging.DEBUG)
            logging.getLogger("mau.client.crypto").setLevel(logging.DEBUG)
            assert await adapter.connect(),"Matrix required encryption connect failed"
            assert adapter._client.crypto is not None and adapter._encryption
            adapters[name]=adapter
            return adapter
    try:
        bot,peer=await connect("bot"),await connect("peer")
        assert bot._crypto_db_path!=peer._crypto_db_path
        for name in ("bot","peer"):
            recovery=scratch/"home"/name/"recovery.txt"
            assert recovery.is_file() and recovery.stat().st_mode&0o777==0o600,"Cross-signing bootstrap failed"
        async def peer_has_text():return any(event.text=="Encrypted original Matrix outbound" for event in received["peer"])
        with profile("bot"):
            sent=await bot.send(room,"Encrypted original Matrix outbound")
            assert sent.success,sent
        await until(peer_has_text)
        async def bot_has_text():return any(event.text=="Encrypted original Matrix reply" for event in received["bot"])
        with profile("peer"):
            sent=await peer.send(room,"Encrypted original Matrix reply")
            assert sent.success,sent
        await until(bot_has_text)
        # Room policy rotates after two messages. The third send creates a new
        # signed Megolm session and shares it through the retained Olm channel.
        with profile("bot"):
            assert (await bot.send(room,"Before rotation")).success
            assert (await bot.send(room,"After rotation")).success
        # Preserve the adapter's normal batching of adjacent text events.
        async def rotated():return any("After rotation" in event.text.splitlines() for event in received["peer"])
        await until(rotated)
        bot_events=[event for _,event in server.events
                    if event["sender"]=="@bot:example.invalid" and event["type"]=="m.room.encrypted"]
        assert bot_events[0]["content"]["session_id"]==bot_events[1]["content"]["session_id"]
        assert bot_events[1]["content"]["session_id"]!=bot_events[2]["content"]["session_id"]
        key_before=bot._client.crypto.account.identity_key
        recovery=(scratch/"home/bot/recovery.txt").read_text().strip()
        await bot.disconnect()
        # Exercise the original pre-device-ID migration with real persisted
        # account, Olm and both Megolm session formats, not synthetic blobs.
        from standalone import matrix_olm
        current,legacy="@bot:example.invalid:BOT","@bot:example.invalid:default"
        session_types={"crypto_olm_session":matrix_olm.Session,
                       "crypto_megolm_inbound_session":matrix_olm.InboundGroupSession,
                       "crypto_megolm_outbound_session":matrix_olm.OutboundGroupSession}
        migrated={}
        with sqlite3.connect(bot._crypto_db_path) as database:
            account,=database.execute("SELECT account FROM crypto_account WHERE account_id=?",
                                      ("@bot:example.invalid",)).fetchone()
            account=matrix_olm.Account.from_pickle(account,current)
            database.execute("UPDATE crypto_account SET account=? WHERE account_id=?",
                             (account.pickle(legacy),"@bot:example.invalid"))
            for table,kind in session_types.items():
                rows=database.execute("SELECT session_id,session FROM "+table+" WHERE account_id=?",
                                      ("@bot:example.invalid",)).fetchall()
                assert rows,"Fixture did not persist "+table
                migrated[table]=len(rows)
                for identifier,pickled in rows:
                    session=kind.from_pickle(pickled,current)
                    database.execute("UPDATE "+table+" SET session=? WHERE account_id=? AND session_id=?",
                                     (session.pickle(legacy),"@bot:example.invalid",identifier))
        # Observe the real migration before the initial membership sync
        # legitimately invalidates the resumed outbound group session.
        from unittest.mock import patch
        original_sweep=matrix.MatrixAdapter._repickle_crypto_sessions
        async def checked_sweep(adapter,database,account_id,old_key,new_key):
            await original_sweep(adapter,database,account_id,old_key,new_key)
            for table,kind in session_types.items():
                rows=await database.fetch("SELECT session FROM "+table+" WHERE account_id=$1",account_id)
                assert len(rows)==migrated[table],table
                for row in rows:kind.from_pickle(row["session"],current)
            report["legacy_pickle_migration"]=True
        with patch.object(matrix.MatrixAdapter,"_repickle_crypto_sessions",checked_sweep):
            bot=await connect("bot",recovery)
        assert bot._client.crypto.account.identity_key==key_before,"Persistent Olm identity changed on restart"
        assert bot._client.crypto._cross_signing_private_keys is not None,"Recovery key did not restore cross-signing"
        assert report["legacy_pickle_migration"],"Original account/session migration was bypassed"
        with profile("bot"):
            assert (await bot.send(room,"Encrypted after restart")).success
            document=scratch/"work/encrypted-fixture.txt"
            document.write_text("Matrix encrypted attachment content")
            result=await bot.send_document(room,str(document),caption="Encrypted document")
            assert result.success,result
        async def attachment():return any(event.media_urls for event in received["peer"])
        await until(attachment)
        media_event=next(event for event in received["peer"] if event.media_urls)
        assert pathlib.Path(media_event.media_urls[0]).read_text()=="Matrix encrypted attachment content"
        assert all(event["type"] in {"m.room.encrypted","m.reaction"} for _,event in server.events)
        assert all("body" not in event["content"] for _,event in server.events)
        assert all(b"Matrix encrypted attachment content" not in value for value in server.media.values())
        assert not any(name in sys.modules for name in ("olm","_libolm","Crypto","Cryptodome","cryptography","asyncpg"))
        report.update(original_connect=True,required_encryption=True,olm_key_exchange=True,megolm_messages=True,
            profile_scope=True,rotation=True,persistent_identity=True,cross_signing_bootstrap=True,
            recovery_key_restore=True,legacy_pickle_migration=True,encrypted_attachment=True,no_plaintext_wire=True,
            requests=len(server.records),encrypted_events=sum(event["type"]=="m.room.encrypted" for _,event in server.events))
    finally:
        report["received"]={name:[event.text for event in values] for name,values in received.items()}
        for adapter in adapters.values():await adapter.disconnect()
        await runner.cleanup()
try:
    asyncio.run(exercise())
except BaseException:
    report["error"]=traceback.format_exc()+"\n"+fixture_logs.getvalue()[-20000:]
report["blocked_events"]=guard.events
print(json.dumps(report),flush=True)
os._exit(0)
'''


@unittest.skipUnless(sys.platform.startswith("linux") and sys.version_info >= (3, 14),
                     "Original Matrix encryption currently verified on Linux/Python 3.14")
class OriginalMatrixEncryptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        platform_fixture.OriginalPlatformSDKTests.setUpClass.__func__(cls)

    def test_original_adapters_olm_megolm_rotation_cross_signing_restart_and_encrypted_file(self):
        report,_=platform_fixture.OriginalPlatformSDKTests.invoke(self,_MATRIX)
        self.assertTrue(all(report[name] for name in ("original_connect","required_encryption","olm_key_exchange",
            "megolm_messages","profile_scope","rotation","persistent_identity","cross_signing_bootstrap",
            "recovery_key_restore","legacy_pickle_migration","encrypted_attachment","no_plaintext_wire")))
        self.assertGreaterEqual(report["encrypted_events"],6)


if __name__=="__main__":
    unittest.main()
