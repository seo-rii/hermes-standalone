"""Portable raw-ABI/policy tests; Windows execution is never simulated as proof."""
import ctypes as C
from pathlib import Path
import os
import tempfile
from types import MethodType, SimpleNamespace
import unittest

from standalone import relay_atif_windows as module
from standalone.relay_files_windows import OBJECT_ATTRIBUTES, WindowsRelayFiles
from standalone.windows_files import BY_HANDLE_FILE_INFORMATION, WindowsPrivateFiles


def value(handle):
    return handle.value if isinstance(handle,C.c_void_p) else handle


class NativeCallRecorder:
    """Exercises the declared ABI with owned buffers, without invoking an OS."""
    def __init__(self, *, targets=('absent','absent'), failure=None, collisions=0):
        self.events=[];self.handles={};self.closed=[];self.targets=list(targets)
        self.failure=failure;self.collisions=collisions;self.serial=100;self.create_calls=[]
        self.discarded=[];self.renamed=None;self.written=bytearray();self.descriptors=[];self.freed=[]
        private=SimpleNamespace(user_string='S-1-5-21-11-22-33-1001')
        private._error=lambda operation,code=None:OSError('owned '+operation+' failure')
        private.descriptor_from_string=self.descriptor
        private._descriptor=MethodType(WindowsPrivateFiles._descriptor,private)
        private.free=lambda descriptor:self.freed.append(value(descriptor))
        private.create_file=self.anchor
        private.file_info=self.file_info
        private.final_path=self.final_path
        private.check_handle=self.check_handle
        private.close=self.close
        private.write=self.write
        private.flush=self.flush
        self.private=private
        self.relay=SimpleNamespace(private=private,create=self.create,status_error=self.status_error,restrict=self.restrict)
        self.relay.validate=MethodType(WindowsRelayFiles.validate,self.relay)
        self.api=object.__new__(module.WindowsAtifAtomic)
        self.api.relay,self.api.private=self.relay,private
        self.api.file_type=self.file_type
        self.api.set_information=self.set_information

    def fresh(self,**state):
        self.serial+=1;self.handles[self.serial]=state;return self.serial

    def anchor(self,path,access,share,security,creation,flags,template):
        self.events.append(('anchor',path,access,share,flags))
        return self.fresh(name=path,directory=True,attributes=0x10,file_type=1)

    def final_path(self,handle,buffer,length,flags):
        self.events.append(('final_path',value(handle)))
        path=self.handles[value(handle)]['name']
        if buffer is None:return len(path)+1
        buffer.value=path;return len(path)

    def descriptor(self,text,revision,out,length):
        self.descriptors.append(text)
        C.cast(out,C.POINTER(C.c_void_p)).contents.value=999
        return True

    def create(self,out,access,attributes,status,allocation,file_attributes,share,disposition,options,ea,ea_length):
        attributes=C.cast(attributes,C.POINTER(OBJECT_ATTRIBUTES)).contents
        text=C.string_at(attributes.ObjectName.contents.Buffer,attributes.ObjectName.contents.Length).decode('utf-16-le','surrogatepass')
        parent=attributes.RootDirectory
        call=dict(parent=parent,name=text,access=access,disposition=disposition,options=options,share=share,
                  descriptor=attributes.SecurityDescriptor)
        self.create_calls.append(call);self.events.append(('create',disposition,parent,text))
        if disposition==module.FILE_OPEN:
            target=self.targets.pop(0) if self.targets else 'absent'
            if target=='absent':return C.c_int32(0xc0000034).value
            attrs=0x10 if target=='directory' else 0x400 if target=='reparse' else 0x80
            handle=self.fresh(name=text,parent=parent,directory=target=='directory',attributes=attrs,
                              file_type=3 if target=='pipe' else 1)
        elif disposition==module.FILE_CREATE:
            if self.collisions:
                self.collisions-=1;return C.c_int32(0xc0000035).value
            handle=self.fresh(name=text,parent=parent,directory=False,attributes=0x80,file_type=1,temporary=True)
        else:
            handle=self.fresh(name=text,parent=parent,directory=True,attributes=0x10,file_type=1)
        C.cast(out,C.POINTER(C.c_void_p)).contents.value=handle
        return 0

    def status_error(self,status):
        return {0xc0000034:2,0xc0000035:183,0xc0000022:5}.get(status&0xffffffff,50)

    def file_info(self,handle,out):
        info=C.cast(out,C.POINTER(BY_HANDLE_FILE_INFORMATION)).contents
        info.dwFileAttributes=self.handles[value(handle)]['attributes'];return True

    def file_type(self,handle):return self.handles[value(handle)]['file_type']
    def restrict(self,handle,flags,descriptor):
        self.events.append(('restrict',value(handle),flags,value(descriptor)));return True
    def check_handle(self,handle):
        handle=value(handle);self.events.append(('acl',handle))
        if self.failure=='acl' and self.handles[handle].get('temporary'):raise OSError('owned ACL refusal')
    def close(self,handle):
        handle=value(handle);self.events.append(('close',handle));self.closed.append(handle);return True
    def write(self,handle,buffer,size,out,overlapped):
        self.events.append(('write',value(handle),size))
        if self.failure=='write':return False
        count=0 if self.failure=='zero_write' else size+1 if self.failure=='overflow_write' else min(size,4093)
        C.cast(out,C.POINTER(module.DWORD)).contents.value=count
        if 0<count<=size:self.written.extend(C.string_at(buffer,count))
        return True
    def flush(self,handle):
        self.events.append(('flush',value(handle)));return self.failure!='flush'
    def set_information(self,handle,status,buffer,length,kind):
        handle=value(handle);self.events.append(('set_information',kind,handle))
        if kind==module.FILE_DISPOSITION_INFORMATION_CLASS:
            information=C.cast(buffer,C.POINTER(module.FILE_DISPOSITION_INFORMATION)).contents
            assert information.DeleteFile and length==1
            self.discarded.append(handle);return 0
        information=C.cast(buffer,C.POINTER(module.FILE_RENAME_INFORMATION)).contents
        text=C.string_at(C.addressof(information)+module.FILE_RENAME_INFORMATION.FileName.offset,information.FileNameLength).decode('utf-16-le','surrogatepass')
        assert information.ReplaceIfExists and length>=C.sizeof(module.FILE_RENAME_INFORMATION)+information.FileNameLength
        if self.failure=='rename':return C.c_int32(0xc0000022).value
        self.renamed=(handle,information.RootDirectory,text)
        assert all(parent not in self.closed for parent,state in self.handles.items() if state['directory'])
        return 0


class WindowsAtomicTests(unittest.TestCase):
    def test_portable_rename_buffer_uses_windows_utf16_and_pointer_alignment(self):
        name='한글-😀.json';buffer=module._rename_information(0x1234,name)
        info=module.FILE_RENAME_INFORMATION.from_buffer(buffer)
        self.assertEqual(info.RootDirectory,0x1234)
        self.assertEqual(info.FileNameLength,len(name.encode('utf-16-le')))
        self.assertEqual(C.string_at(C.addressof(buffer)+module.FILE_RENAME_INFORMATION.FileName.offset,info.FileNameLength),name.encode('utf-16-le'))
        pointer=C.sizeof(C.c_void_p)
        self.assertEqual(module.FILE_RENAME_INFORMATION.RootDirectory.offset,pointer)
        self.assertEqual(module.FILE_RENAME_INFORMATION.FileNameLength.offset,2*pointer)
        self.assertEqual(module.FILE_RENAME_INFORMATION.FileName.offset,2*pointer+4)
        self.assertEqual(C.sizeof(module.FILE_DISPOSITION_INFORMATION),1)
        for invalid in ('','..','.','a/b','a\\b','a\0b','output:stream','C:relative'):
            with self.subTest(invalid=repr(invalid)),self.assertRaises(OSError):module._rename_information(1,invalid)
        if os.name!='nt':
            with self.assertRaisesRegex(OSError,'requires a Windows host'):module.WindowsAtifAtomic()

    def test_handle_relative_walk_partial_write_flush_recheck_atomic_rename_and_owner_policy(self):
        recorder=NativeCallRecorder(targets=('regular','regular'))
        payload=(b'owned\0payload\n'*13000)+b'end'
        recorder.api.write(r'C:\trusted',['private','nested'],'한글.json',payload)
        self.assertEqual(recorder.written,payload)
        self.assertIsNotNone(recorder.renamed)
        self.assertEqual(recorder.renamed[2],'한글.json')
        self.assertEqual(len([event for event in recorder.events if event[0]=='anchor']),1)
        self.assertEqual(len([event for event in recorder.events if event[0]=='final_path']),2)
        self.assertEqual(recorder.discarded,[])
        self.assertEqual(sorted(recorder.closed),sorted(recorder.handles))
        self.assertEqual(len(recorder.closed),len(set(recorder.closed)))
        self.assertEqual(len(recorder.freed),len(recorder.descriptors))
        for descriptor in recorder.descriptors:
            self.assertEqual(descriptor,'O:S-1-5-21-11-22-33-1001D:P(A;;FA;;;S-1-5-21-11-22-33-1001)(A;;FA;;;SY)')
        children=[call for call in recorder.create_calls if call['disposition']==module.FILE_OPEN_IF]
        self.assertEqual([call['name'] for call in children],['private','nested'])
        self.assertTrue(all(call['share']==3 and call['options']&module.FILE_OPEN_REPARSE_POINT for call in children))
        temporary=next(call for call in recorder.create_calls if call['disposition']==module.FILE_CREATE)
        self.assertEqual(temporary['parent'],recorder.renamed[1]);self.assertEqual(temporary['share'],0)
        events=recorder.events
        initial=next(index for index,event in enumerate(events) if event[:2]==('create',module.FILE_OPEN))
        flush=next(index for index,event in enumerate(events) if event[0]=='flush')
        second=next(index for index,event in enumerate(events) if event[:2]==('create',module.FILE_OPEN) and index>initial)
        rename=next(index for index,event in enumerate(events) if event[:2]==('set_information',10))
        self.assertLess(initial,flush);self.assertLess(flush,second);self.assertLess(second,rename)

    def test_nonregular_or_reparse_target_refusal_before_and_after_temporary_write(self):
        for kind in ('directory','reparse','pipe'):
            for stage in ('before','after'):
                with self.subTest(kind=kind,stage=stage):
                    recorder=NativeCallRecorder(targets=(kind,) if stage=='before' else ('absent',kind))
                    with self.assertRaises(OSError):recorder.api.write(r'C:\trusted',[],'output.json',b'private')
                    self.assertIsNone(recorder.renamed)
                    self.assertEqual(len(recorder.discarded),int(stage=='after'))
                    self.assertEqual(sorted(recorder.closed),sorted(recorder.handles))
                    self.assertEqual(bool(recorder.written),stage=='after')

    def test_exclusive_temporary_retries_and_failure_cleanup_keep_no_successful_commit(self):
        recorder=NativeCallRecorder(collisions=3)
        recorder.api.write(r'C:\trusted',[],'output.json',b'owned')
        names=[call['name'] for call in recorder.create_calls if call['disposition']==module.FILE_CREATE]
        self.assertEqual(len(names),4);self.assertEqual(len(set(names)),4)
        for failure in ('write','zero_write','overflow_write','flush','rename','acl'):
            with self.subTest(failure=failure):
                recorder=NativeCallRecorder(failure=failure)
                with self.assertRaises(OSError):recorder.api.write(r'C:\trusted',[],'output.json',b'owned')
                self.assertEqual(len(recorder.discarded),1)
                self.assertIsNone(recorder.renamed)
                self.assertEqual(sorted(recorder.closed),sorted(recorder.handles))
        recorder=NativeCallRecorder(collisions=16)
        with self.assertRaisesRegex(OSError,'could not allocate'):recorder.api.write(r'C:\trusted',[],'output.json',b'owned')
        self.assertEqual(sum(call['disposition']==module.FILE_CREATE for call in recorder.create_calls),16)
        self.assertEqual(recorder.discarded,[])
        self.assertEqual(sorted(recorder.closed),sorted(recorder.handles))

    @unittest.skipUnless(os.name=='nt','Actual Windows kernel and ACL proof requires a Windows host')
    def test_native_windows_private_atomic_replacement_and_reparse_refusal(self):
        from standalone.windows_files import _api,constants
        with tempfile.TemporaryDirectory(prefix='hermes-atif-native-') as name:
            anchor=Path(name).resolve()
            module.atomic_write(anchor,['private','nested'],'output.json',b'first\0payload')
            path=anchor/'private/nested/output.json'
            self.assertEqual(path.read_bytes(),b'first\0payload')
            module.atomic_write(anchor,['private','nested'],'output.json',b'second')
            self.assertEqual(path.read_bytes(),b'second')
            api=_api();handle=api.open_private(path,constants.GENERIC_READ|constants.READ_CONTROL,3,0x80,7)
            api.close(handle)
            target=anchor/'directory';target.mkdir()
            with self.assertRaises(OSError):module.atomic_write(anchor,[],'directory',b'blocked')
            self.assertEqual(list(anchor.glob('.*.tmp')),[])
            outside=anchor/'outside';outside.mkdir()
            try:os.symlink(outside,anchor/'trap',target_is_directory=True)
            except OSError as error:
                if getattr(error,'winerror',None)==1314:return
                raise
            with self.assertRaises(OSError):module.atomic_write(anchor,['trap'],'blocked.json',b'blocked')
            self.assertFalse((outside/'blocked.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
