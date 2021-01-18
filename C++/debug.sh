rm debug/*
rm fixBug/*

for i in {1..30}
do
    ./a.out >> ./debug/$i.txt
done
