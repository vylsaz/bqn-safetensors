# libcbqn substitute for windows

This is a libcbqn substitute (hack) to get this library to work on Windows with BQN.exe.

## build

using `gcc` and `ar` from mingw (ucrt)
```
gcc -O2 -c cbqn.c -o cbqn.o 
ar rcs libcbqn.a cbqn.o 
```

## caveats

Because on Windows a DLL cannot be linked against and use the functions in an EXE, this "libcbqn" gets the functions by finding the handle of current process (assumed to be BQN.exe) and get the function pointers.

Because of this, any DLL that uses this hack can only be used by BQN.exe. 

This also requires that BQN.exe exports the functions we need, which it does for now.
