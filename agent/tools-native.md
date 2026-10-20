Your one file tool is the shell, `bash`: each call runs one command in your working directory, in a
new shell, and returns its exit code and output. View a file with `cat` or `sed -n '1,80p' f`, create
one with `cat > f <<'EOF'`, and change one by rewriting it the same way or with a short `python3`
script. The shell has the judge's compilers (`gcc`, `g++`, `gfortran`), `python3` and binutils, so
check every rewrite locally. The benchmark tools above reach you through the MCP server under exactly
those names.
