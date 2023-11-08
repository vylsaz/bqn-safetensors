lib ← "./target/release/"∾"bqn_safetensors"{
    𝕨𝕊"windows": 𝕨∾".dll"; 
    "lib"∾𝕨∾".so"
}•platform.os

("'"∾lib∾"' not found") ! •file.Exists lib

SaveFile ⇐ {
    f ← lib •FFI "a"‿"serialize"‿"a"‿"a"
    x ← ⥊𝕩
    b ← 0=•internal.ElType¨x
    x "Ai8"⊸•internal.Variation¨⌾(b/⊢) ↩
    ⟨k,r⟩ ← F⟨𝕨,x⟩
    {k≡0? !r ; r}
}

OpenFile ⇐ {𝕊file:
    f ← lib •FFI "a"‿"get_names"‿"a"
    ⟨k,r⟩ ← F⟨file⟩
    {k≡0? 
        !r 
    ;
        f ← lib •FFI "a"‿"get_tensor"‿"a"‿"a"
        {
            file ⇐ file
            keys ⇐ r
            Get ⇐ {
                ⟨k,r⟩ ← F⟨file,𝕩⟩
                {k≡0? !r ; r}
            }
        }
    }
}
