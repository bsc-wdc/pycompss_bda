

for file in $(ls ../src/kmeans*.py); do
    echo "Reporting complexities for $file"
    cloc $file
    bblfsh-tools cyclomatic $file
    bblfsh-tools npath $file | grep FuncName | grep -v main | awk '{print $2}' | tr ':' '\t' | awk '{print $2}' > out
    python compute_npaths.py out
done

for file in $(ls ../src/csvm*.py); do
    echo "Reporting complexities for $file"
    cloc $file
    bblfsh-tools cyclomatic $file
    bblfsh-tools npath $file | grep FuncName | grep -v main | awk '{print $2}' | tr ':' '\t' | awk '{print $2}' > out
    python compute_npaths.py out
done

