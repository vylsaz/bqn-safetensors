lib ← •args •file.At "bqn_safetensors"{
    𝕨𝕊"windows": 𝕨∾".dll"; 
    "lib"∾𝕨∾".so"
}•platform.os

("'"∾lib∾"' not found") ! •file.Exists lib

ffi ← {
    serialize ⇐ lib •FFI "a"‿"serialize"‿"a"‿"a"‿"a"
    getNames ⇐ lib •FFI "a"‿"get_names"‿"a"
    getTensor ⇐ lib •FFI "a"‿"get_tensor"‿"a"‿"a"
}

SaveFile ⇐ {𝕨𝕊x:
    k‿v ← {
    0: <˘⍉x ;
    6: ⟨x.Keys @, x.Values @⟩ ;
        !"Wrong type for argument 𝕩"
    }•Type x
    k •ToUTF8¨↩
    b ← 0=•internal.ElType¨v
    v "Ai8"⊸•internal.Variation¨⌾(b⊸/) ↩
    ⟨ok,r⟩ ← ffi.Serialize ⟨•ToUTF8 𝕨,k,v⟩
    {ok≡0? !r ; r}
}

LoadFile ⇐ {𝕊file:
    file (•ToUTF8 •file.RealPath)↩
    ⟨ok,r⟩ ← ffi.GetNames ⟨file⟩
    {ok≡0? !r ;
        {
            dict ← r •HashMap @¨r
            file ← file
            Get ⇐ {𝕊k:
                v ← dict.Get k
                {v≢@? v ;
                    ⟨ok,r⟩ ← ffi.GetTensor⟨file,•ToUTF8 k⟩
                    {ok≡0? !r ; r⊣k dict.Set r}
                }
            }
            Has ⇐ dict.Has
            Keys ⇐ dict.Keys
            Values ⇐ Get¨Keys
            Count ⇐ dict.Count
        }
    }
}
