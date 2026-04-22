import pefile
import sys

try:
    pe = pefile.PE(r"C:\Users\Wren C\Documents\Coding stuff\Vault Reconstruct\reconstruct_rust\target\debug\build\getrandom-25c396aa4d514c2e\build-script-build.exe")
    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        print(entry.dll.decode('utf-8'))
except Exception as e:
    print(f"Error: {e}")
