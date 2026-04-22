package main

/*
#include <stdlib.h>
*/
import "C"
import (
    "io"
    "os"
    "unsafe"

    "github.com/parquet-go/parquet-go"
)

type Row struct {
    Text string `parquet:"text"`
}

var currentReader *parquet.Reader
var currentFile *os.File

//export LoadShard
func LoadShard(path *C.char) C.int {
    p := C.GoString(path)
    if currentFile != nil {
        currentFile.Close()
    }

    f, err := os.Open(p)
    if err != nil {
        return -1
    }
    currentFile = f
    currentReader = parquet.NewReader(f)
    return C.int(currentReader.NumRows())
}

//export GetBatch
func GetBatch(batchSize int) **C.char {
    if currentReader == nil {
        return nil
    }

    rows := make([]Row, batchSize)
    n := 0
    for i := 0; i < batchSize; i++ {
        err := currentReader.Read(&rows[i])
        if err != nil {
            if err == io.EOF {
                break
            }
            return nil
        }
        n++
    }

    if n == 0 {
        return nil
    }

    // Allocate C array of strings
    ptrSize := unsafe.Sizeof((*C.char)(nil))
    ptr := C.malloc(C.size_t(batchSize) * C.size_t(ptrSize))
    cStrings := (*[1 << 30]*C.char)(ptr)

    for i := 0; i < n; i++ {
        cStrings[i] = C.CString(rows[i].Text)
    }
    for i := n; i < batchSize; i++ {
        cStrings[i] = nil
    }

    return (**C.char)(ptr)
}

//export FreeBatch
func FreeBatch(batch **C.char, size int) {
    if batch == nil {
        return
    }
    cStrings := (*[1 << 30]*C.char)(unsafe.Pointer(batch))
    for i := 0; i < size; i++ {
        if cStrings[i] != nil {
            C.free(unsafe.Pointer(cStrings[i]))
        }
    }
    C.free(unsafe.Pointer(batch))
}

func main() {}
