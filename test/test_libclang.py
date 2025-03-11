import clang.cindex

# location of llvm
clang.cindex.Config.set_library_file("/opt/homebrew/opt/llvm/lib/libclang.dylib")

index = clang.cindex.Index.create()
tu = index.parse("/my_project/example.cpp", args=["-std=c++17"])

print("Translation Unit:", tu.spelling)
for child in tu.cursor.get_children():
    print(child.kind, child.spelling)
