a=1
for i in *.png
do
	convert $i sample_road_img_$a.png
	a=$((a+1))
done
