"""eSpeak GetTranslatedPhonemeString for the actual Piper ASCII/IPA modes."""
from standalone.local_piper_phoneme_ipa import IPA


def write_phone(vm,phone,sequence,index,use_ipa):
    if phone.code==15:
        return ''
    if phone.code==21:
        return '('+vm.data.tables[sequence[index].tone_ph].name+')'
    if use_ipa:
        name=vm.run(sequence,index,start=index).ipa_string
        if name.startswith(' '):
            return ''
        if name and ord(name[0])<32:
            name=name[1:]
        if name:
            return name
    result=[];first=True
    for code in phone.mnemonic.to_bytes(4,'little'):
        if code==0 or code==47:
            break
        if use_ipa:
            if first and code==95 or code==35 and phone.type==2:
                break
            if not first and 48<=code<=57:
                continue
            if 32<=code<128:
                code=IPA[code-32]
        result.append(chr(code));first=False
    return ''.join(result)


def render(vm,sequence,mode=2):
    if mode not in (0,2):
        raise ValueError('This Piper boundary consumes only ASCII or IPA phoneme modes')
    use_ipa=bool(mode&2);output=[]
    for index in range(1,len(sequence)-2):
        item=sequence[index]
        name=write_phone(vm,item.ph,sequence,index,use_ipa)
        if item.newword&1 and not item.newword&12:
            output.append(' ')
        if item.synthflags&4 and item.stresslevel>1:
            stress=min(item.stresslevel,5)
            output.append(('ˈ' if stress>3 else 'ˌ') if use_ipa else "==,,''"[stress])
        output.append(name)
        if item.ph.code!=21:
            if item.synthflags&8:
                output.append(write_phone(vm,vm.phones[12],sequence,index,use_ipa))
            if item.synthflags&4 and item.type!=2:
                output.append(write_phone(vm,vm.phones[20],sequence,index,use_ipa))
            if item.tone_ph>0:
                output.append(write_phone(vm,vm.phones[item.tone_ph],sequence,index,use_ipa))
    return ''.join(output)
