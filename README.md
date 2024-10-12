# bqn-safetensors
BQN safetensors binding to load and store arrays in that format

## build
> **_For Windows_**: build windows/cbqn.c first, see [this](windows/README.md).

```
cargo build --release
```
## usage
```
# dir: directory where the shared lib is
# no dir = current directory
SaveFile‿LoadFile ← dir •Import "./safetensors.bqn"

# saving
"./test.safetensors" SaveFile ["a"‿(2‿2⥊0), "b"‿(2‿3‿4⥊↕24)]

# loading
loaded ← LoadFile "./test.safetensors"
loaded.Get "b"
```
```
r ← w SaveFile x
```
`w`: file name  
`x`: either an n-by-2 array, or a namespace with fields `Keys` and `Values`  
`r`: `@`

```
r ← LoadFile x
```
`x`: file name  
`r`: a namespace with fields `Get`, `Has`, `Count`, `Keys` and `Values`
