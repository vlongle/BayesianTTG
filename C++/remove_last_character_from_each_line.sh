# Argument: filename. $1 in bash.
# https://www.cyberciti.biz/faq/sed-remove-last-character-from-each-line/

sed -i "" "s/.$//" $1
