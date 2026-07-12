import pefile
import sys

try:
    pe = pefile.PE(sys.argv[1])
    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        print(entry.dll.decode('utf-8'))
except Exception as e:
    print(f"Error: {e}")
