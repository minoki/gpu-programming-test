@if defined INTELOCLSDKROOT (
    cl /nologo /I"%INTELOCLSDKROOT%\include" list.c /link /LIBPATH:"%INTELOCLSDKROOT%\lib\x64" OpenCL.lib
    cl /nologo /I"%INTELOCLSDKROOT%\include" add.c /link /LIBPATH:"%INTELOCLSDKROOT%\lib\x64" OpenCL.lib
) else (
    if defined CUDA_PATH (
        cl /nologo /I"%CUDA_PATH%\include" list.c /link /LIBPATH:"%CUDA_PATH%\lib\x64" OpenCL.lib
        cl /nologo /I"%CUDA_PATH%\include" add.c /link /LIBPATH:"%CUDA_PATH%\lib\x64" OpenCL.lib
    ) else (
        echo OpenCL SDK not found
    )
)
