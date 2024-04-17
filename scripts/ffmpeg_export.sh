ffmpeg -r 60 -i pred/frame_%04d.png -pix_fmt yuv420p -b:v 50M  pred.mp4
