# for i in {100, 150, 200, 500, 1000, 2000}; do taskset -c 0 ./query 128 1000 "$i" | sed -n 9p; done
# for i in {100, 150, 200, 500, 1000, 2000}; do taskset -c 0 ./query 255 1000 "$i" | sed -n 9p; done
# for i in {100, 150, 200, 500, 1000, 2000}; do taskset -c 0 ./query 512 1000 "$i" | sed -n 9p; done
# for i in {100, 150, 200, 500, 1000, 2000}; do taskset -c 0 ./query 1024 1000 "$i" | sed -n 9p; done
# for i in {100, 150, 200, 500, 1000, 2000}; do taskset -c 0 ./query 2048 1000 "$i" | sed -n 9p; done
# for i in { 100, 150, 200, 500, 1000, 2000}; do taskset -c 0 ./query 4096 1000 "$i" | sed -n 9p; done
# for i in { 100, 150, 200, 500, 1000, 2000}; do taskset -c 0 ./query 8192 1000 "$i" | sed -n 9p; done
# for i in { 100, 150, 200, 500, 1000, 2000}; do taskset -c 0 ./query 16384 1000 "$i" | sed -n 9p; done
# for i in { 128, 256, 512, 1024, 2048, 4096, 8192, 16384}; do ./index "$i"; done
echo "Mode1"
for i in 100 200 300 500 700 1000 1500 2000 3000; do taskset -c 0 ./query sift 1024 3 "$i" 1 | sed -n 7p; done
# echo "Mode2"
# for i in 1000 2000 3000 5000 10000 30000 100000; do taskset -c 0 ./query sift 1024 3 "$i" 2 | sed -n 7p; done
# echo "Mode3"
# for i in 1000 2000 3000 4000 5000 7000 10000; do taskset -c 0 ./query sift 1024 3 "$i" 3 | sed -n 7p; done

