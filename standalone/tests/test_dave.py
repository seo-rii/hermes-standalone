"""Official MLS vectors and independent original davey wire recordings."""

from pathlib import Path
import unittest

from standalone import vendor
from standalone.tests import test_crypto_google as crypto_fixture


FIXTURES = Path(__file__).resolve().with_name("fixtures") / "dave"


class DaveTests(unittest.TestCase):
    DEPENDENCIES = ("tlslite-ng", "ecdsa")

    @classmethod
    def setUpClass(cls):
        crypto_fixture.GoogleAuthCryptoTests.setUpClass.__func__(cls)
        packages = {item["name"]: item for item in vendor.load_dependencies(crypto_fixture.PROJECT / "standalone/test-dependencies.json")["packages"]}
        pending, selected = ["pytest"], set()
        while pending:
            name = pending.pop()
            if name in selected:
                continue
            selected.add(name)
            pending.extend(packages[name]["dependencies"])
        for name in selected:
            files, _ = vendor.read_wheel(packages[name], crypto_fixture.DOWNLOADS)
            for path, content in files.items():
                target = cls.library / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)

    def execute(self, program):
        prefix = f"fixture_root=Path({str(FIXTURES)!r})\n"
        prefix += "import json\ndata=json.loads((fixture_root/'dave-mls.json').read_text())\n"
        crypto_fixture.GoogleAuthCryptoTests.execute(self, prefix + program)

    def test_rfc9180_p256_hpke_all_257_ciphertexts_export_and_authentication(self):
        self.execute('''
from standalone.mls_crypto import *
from standalone.crypto import InvalidTag
for case in data['hpke']:
    b=lambda k:bytes.fromhex(case[k])
    assert derive_key_pair(b('ikmR'))==(b('skRm'),b('pkRm'))
    assert derive_key_pair(b('ikmE'))==(b('skEm'),b('pkEm'))
    enc,sender=setup_sender(b('pkRm'),b('info'),private=b('skEm'))
    receiver=setup_receiver(b('skRm'),enc,b('info'))
    assert enc==b('enc')
    for name in ('key','base_nonce','exporter_secret'):
        assert getattr(sender,name)==b(name)
    for item in case['encryptions']:
        plain,aad,cipher=(bytes.fromhex(item[k]) for k in ('pt','aad','ct'))
        assert sender.seal(plain,aad)==cipher
        corrupt=bytearray(cipher);corrupt[-1]^=1
        try:receiver.open(corrupt,aad)
        except InvalidTag:pass
        else:raise AssertionError('HPKE released unauthenticated plaintext')
        assert receiver.open(cipher,aad)==plain
    for item in case['exports']:
        assert sender.export(bytes.fromhex(item['exporter_context']),item['L'])==bytes.fromhex(item['exported_value'])
for private in (bytes(31),bytes(32),P256_ORDER.to_bytes(32,'big')):
    try:public_key(private)
    except ValueError:pass
    else:raise AssertionError('Malformed P-256 private key accepted')
for public in (bytes(65),b'\\x04'+bytes(64),b'\\x02'+bytes(32)):
    try:setup_sender(public,b'')
    except ValueError:pass
    else:raise AssertionError('Malformed P-256 recipient accepted')
assert not any(name in sys.modules for name in ('davey','cryptography','Crypto','nacl'))
''')

    def test_official_mls_suite2_crypto_schedules_and_exporters(self):
        self.execute('''
from standalone.mls_crypto import *
from standalone.mls_schedule import EpochSecrets
from standalone.mls_types import Reader,GroupContext
for case in data['crypto-basics']:
    for field in ('derive_secret','derive_tree_secret','expand_with_label','ref_hash','sign_with_label','encrypt_with_label'):
        item=case[field];b=lambda k:bytes.fromhex(item[k]);label=item['label']
        if field=='derive_secret':assert derive_secret(b('secret'),label)==b('out')
        elif field=='derive_tree_secret':assert expand_label(b('secret'),label,item['generation'].to_bytes(4,'big'),item['length'])==b('out')
        elif field=='expand_with_label':assert expand_label(b('secret'),label,b('context'),item['length'])==b('out')
        elif field=='ref_hash':assert ref_hash(label,b('value'))==b('out')
        elif field=='sign_with_label':
            assert verify_label(b('pub'),label,b('content'),b('signature'))
            assert not verify_label(b('pub'),label,b('content')+b'changed',b('signature'))
        elif field=='encrypt_with_label':assert open_label(b('priv'),label,b('context'),b('kem_output'),b('ciphertext'))==b('plaintext')
for case in data['key-schedule']:
    init=bytes.fromhex(case['initial_init_secret'])
    for item in case['epochs']:
        b=lambda k:bytes.fromhex(item[k]);reader=Reader(b('group_context'));context=GroupContext.read(reader);reader.end()
        joiner=EpochSecrets.joiner(init,b('commit_secret'),context)
        assert joiner==b('joiner_secret')
        assert EpochSecrets.welcome(joiner,b('psk_secret'))==b('welcome_secret')
        secrets=EpochSecrets.from_joiner(joiner,context,b('psk_secret'))
        for name in secrets.LABELS:assert getattr(secrets,name)==b(name)
        assert derive_key_pair(secrets.external_secret)[1]==b('external_pub')
        export=item['exporter']
        assert secrets.export(export['label'],bytes.fromhex(export['context']),export['length'])==bytes.fromhex(export['secret'])
        init=secrets.init_secret
''')

    def test_official_tree_hashes_resolutions_leaf_and_parent_authentication(self):
        self.execute('''
from dataclasses import replace
from standalone.mls_tree import Tree
from standalone.mls_types import LeafNode,ParentNode
for case in data['tree-validation']:
    tree=Tree.parse(bytes.fromhex(case['tree']))
    assert tree.encode().hex()==case['tree']
    assert [tree.tree_hash(index).hex() for index in range(tree.width)]==case['tree_hashes']
    assert [tree.resolution(index) for index in range(tree.width)]==case['resolutions']
    tree.validate(bytes.fromhex(case['group_id']))
    # A forged parent key cannot be accepted merely because the leaves signed.
    index=next(index for index,node in enumerate(tree.nodes) if isinstance(node,ParentNode))
    changed=tree.clone();changed.nodes[index]=replace(changed.nodes[index],parent_hash=b'changed')
    try:changed.validate(bytes.fromhex(case['group_id']))
    except ValueError:pass
    else:raise AssertionError('Forged parent hash accepted')
''')

    def test_official_welcome_and_26_commit_epochs_with_psks_and_treekem(self):
        self.execute('''
from standalone.mls_group import Group
from standalone.mls_types import parse_message
for case in data['passive-client-welcome']+data['passive-client-handling-commit']:
    b=lambda k:bytes.fromhex(case[k]);package=parse_message(b('key_package'))
    psks={(1,bytes.fromhex(item['psk_id']),None,None):bytes.fromhex(item['psk']) for item in case['external_psks']}
    group=Group.join(parse_message(b('welcome')),package,b('init_priv'),b('encryption_priv'),b('signature_priv'),
        tree_data=None if case['ratchet_tree'] is None else b('ratchet_tree'),psks=psks)
    # The official vectors deliberately contain expired reference lifetimes;
    # OpenMLS' own KAT disables lifetime checks. Pick their recorded epoch.
    group.verification_time=package.leaf.lifetime[0]
    assert group.secrets.epoch_authenticator==b('initial_epoch_authenticator')
    for epoch in case['epochs']:
        for proposal in epoch['proposals']:group.queue(bytes.fromhex(proposal))
        before=group.context.encode()
        corrupt=bytearray.fromhex(epoch['commit']);corrupt[-1]^=1
        try:group.merge_commit(corrupt)
        except ValueError:pass
        else:raise AssertionError('Corrupted Commit was accepted')
        assert group.context.encode()==before,'Rejected Commit mutated the active epoch'
        group.merge_commit(bytes.fromhex(epoch['commit']))
        assert group.secrets.epoch_authenticator.hex()==epoch['epoch_authenticator']
''')

    def test_official_treekem_updates_all_receiving_members_and_commit_secrets(self):
        self.execute('''
from standalone.mls_codec import Reader
from standalone.mls_group import _path_key,decrypt_path,merge_path
from standalone.mls_tree import Tree
from standalone.mls_types import GroupContext,UpdatePath
for case in data['treekem']:
    tree=Tree.parse(bytes.fromhex(case['ratchet_tree']))
    for update in case['update_paths']:
        reader=Reader(bytes.fromhex(update['update_path']));path=UpdatePath.read(reader);reader.end()
        changed=tree.clone();merge_path(changed,bytes.fromhex(case['group_id']),update['sender'],path)
        assert changed.tree_hash().hex()==update['tree_hash_after']
        changed.validate(bytes.fromhex(case['group_id']))
        context=GroupContext(1,2,bytes.fromhex(case['group_id']),case['epoch'],changed.tree_hash(),bytes.fromhex(case['confirmed_transcript_hash']),())
        for leaf in case['leaves_private']:
            if leaf['index']==update['sender']:continue
            keys={tree.leaf(leaf['index']).encryption_key:bytes.fromhex(leaf['encryption_priv'])}
            for entry in leaf['path_secrets']:
                private,public=_path_key(bytes.fromhex(entry['path_secret']));keys[public]=private
            secret,new_keys=decrypt_path(changed,update['sender'],path,keys,context)
            assert secret.hex()==update['commit_secret']
''')

    def test_native_dave_ratchet_frame_ciphertexts_replay_and_epoch_grace(self):
        self.execute('''
from standalone.dave_frames import *
for case in data['native']['ratchets']:
    assert HashRatchet(bytes.fromhex(case['secret'])).get(case['generation'])==(bytes.fromhex(case['key']),bytes.fromhex(case['nonce']))
for case in data['native']['frames']:
    secret,plain,cipher=(bytes.fromhex(case[name]) for name in ('secret','plaintext','ciphertext'))
    encryptor=FrameEncryptor(secret);encryptor.truncated_nonce=case['before']
    assert encryptor.encrypt(plain)==cipher
    now=[0.0];decryptor=FrameDecryptor(secret,clock=lambda:now[0]);now[0]=40000000
    assert decryptor.decrypt(cipher)==plain
for case in data['native']['replay']:
    decryptor=FrameDecryptor(bytes.fromhex(case['secret']));observed=[]
    for step in case['sequence']:
        if step.startswith('~'):
            decryptor.transition(bytes.fromhex(step[1:]));observed.append('transition');continue
        try:observed.append(decryptor.decrypt(bytes.fromhex(step)).hex())
        except FrameError:observed.append('refused')
    assert observed==case['expected']
first,second=data['native']['frames'][0],data['native']['frames'][10]
now=[0.0];decryptor=FrameDecryptor(bytes.fromhex(first['secret']),clock=lambda:now[0])
decryptor.transition(bytes.fromhex(second['secret']))
now[0]=10.001
try:decryptor.decrypt(bytes.fromhex(first['ciphertext']))
except FrameError:pass
else:raise AssertionError('Expired epoch ratchet was retained')
assert decryptor.decrypt(bytes.fromhex(second['ciphertext'])).hex()==second['plaintext']
try:decryptor.decrypt(b'plaintext Opus is not an encrypted frame')
except FrameError:pass
else:raise AssertionError('Plaintext bypassed negotiated DAVE')
decryptor.set_passthrough(True);assert decryptor.decrypt(b'negotiated transition')==b'negotiated transition'
decryptor.set_passthrough(False,10);now[0]+=10.001
try:decryptor.decrypt(b'expired negotiated transition')
except FrameError:pass
else:raise AssertionError('Passthrough survived transition expiry')
''')

    def test_native_session_welcome_own_commit_remove_and_rekey_exact_bytes(self):
        self.execute('''
from unittest.mock import patch
from standalone import dave
from standalone.mls_crypto import public_key
from standalone.mls_types import KeyPackage,parse_message
from standalone.mls_codec import opaque
case=data['native_session'];signer=bytes.fromhex(case['signer'])
session=dave.DaveSession(1,123456789012345678,918273645012345678,dave.SigningKeyPair(signer,public_key(signer)))
session.set_external_sender(bytes.fromhex(case['external_sender']))
# Recreate the exact recorded KeyPackage with its public reference-test keys.
blocks=iter([bytes.fromhex(case['init_private']),bytes.fromhex(case['leaf_private'])])
with patch('os.urandom',side_effect=lambda size:next(blocks)):
    assert session.get_serialized_key_package().hex()==case['package']
session.process_welcome(bytes.fromhex(case['initial_welcome']))
assert session.ready and session.status==dave.SessionStatus.active
assert session.get_epoch_authenticator().hex()==case['initial_authenticator']
for uid,expected in case['initial_exports'].items():
    assert session.group.export('Discord Secure Frames v0',int(uid).to_bytes(8,'little'),16).hex()==expected
for epoch in case['epochs']:
    blocks=iter(bytes.fromhex(value) for value in epoch.get('random_blocks',[]))
    if epoch.get('self_update'):
        with patch('os.urandom',side_effect=lambda size:next(blocks)):
            message,welcome=session.group.create_commit(force_path=True)
        assert welcome is None and message.message().hex()==epoch['commit']
    elif epoch.get('random_blocks') is not None:
        with patch('os.urandom',side_effect=lambda size:next(blocks)):
            result=session.process_proposals(dave.ProposalsOperationType.append,opaque(b''.join(bytes.fromhex(value) for value in epoch['proposals'])))
        assert result.commit.hex()==epoch['commit'] and result.welcome.hex()==epoch['welcome']
    else:
        session.process_proposals(dave.ProposalsOperationType.append,opaque(b''.join(bytes.fromhex(value) for value in epoch['proposals'])))
    session.process_commit(bytes.fromhex(epoch['commit']))
    assert session.get_epoch_authenticator().hex()==epoch['authenticator']
assert session.epoch==4 and session.get_user_ids()==['111111111111111111','123456789012345678']
assert session.voice_privacy_code==dave.generate_displayable_code(session.get_epoch_authenticator(),30,5)
assert not any(name in sys.modules for name in ('davey','nacl','Crypto','cryptography'))
''')

    def test_original_davey_binding_regressions_keep_all_assertions(self):
        self.execute('''
count=0
for path in sorted(fixture_root.glob('test_*.py.txt')):
    source=path.read_text()
    assert source.count('import davey')==1
    source=source.replace('import davey','from standalone import dave as davey')
    namespace={'__name__':'dave_reference_'+path.stem}
    exec(compile(source,str(path),'exec'),namespace)
    for name,function in namespace.items():
        if name.startswith('test_') and callable(function):function();count+=1
assert count==32,count
''')

    def test_member_removal_refuses_frames_and_requires_sdk_reinitialization(self):
        self.execute('''
from unittest.mock import patch
from standalone import dave
from standalone.mls_crypto import public_key
from standalone.mls_types import KeyPackage,Proposal
case=data['native_session'];signer=bytes.fromhex(case['signer'])
member=dave.DaveSession(1,123456789012345678,918273645012345678,dave.SigningKeyPair(signer,public_key(signer)))
member.set_external_sender(bytes.fromhex(case['external_sender']))
blocks=iter(bytes.fromhex(case[name]) for name in ('init_private','leaf_private'))
with patch('os.urandom',side_effect=lambda size:next(blocks)):member.get_serialized_key_package()
member.process_welcome(bytes.fromhex(case['initial_welcome']))
joining=dave.DaveSession(1,444444444444444444,member.channel_id)
joining.set_external_sender(bytes.fromhex(case['external_sender']))
package=KeyPackage.parse(joining.get_serialized_key_package())
commit,welcome=member.group.create_commit([Proposal(1,package)])
member.process_commit(commit.message());joining.process_welcome(welcome.encode())
assert joining.ready and joining.own_leaf_index==2
commit,_=member.group.create_commit([Proposal(3,joining.own_leaf_index)])
try:joining.process_commit(commit.message())
except ValueError as error:assert 'removed' in str(error) and 'reinitialization' in str(error)
else:raise AssertionError('Evicted member reported a ready Commit transition')
assert not joining.ready and joining.encryptor is None and not joining.decryptors
try:joining.encrypt_opus(bytes.fromhex(case['media']['plaintext']))
except ValueError:pass
else:raise AssertionError('Evicted member encrypted with an old group key')
joining.reinit(1,joining.user_id,joining.channel_id)
assert not joining.ready and joining.status==dave.SessionStatus.pending
assert joining.get_serialized_key_package()!=package.encode()
''')


if __name__ == "__main__":
    unittest.main()
