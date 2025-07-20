# Project Documentation and Context

Run this command to read all project documentation and context files:

```bash
files=$(find .ai/steering -type f -name "*.md" 2>/dev/null; find . -name "README.md" -not -path "./docs/build/*" 2>/dev/null)
for file in $files; do echo "=== $file ==="; cat "$file"; echo; done
```

This will display all steering documentation and README.md files in the project.

After reading all the files execute:

```bash
git ls-files | tree --fromfile
```

to get to know all files in the project.