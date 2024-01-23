# CuneiML

This repository hosts the code to get the artifects of Cuneiform in the paper CuneiML. We provoide:

- Image: major face cutouts (we provide the bounding boxes only, users need to obtain the orignal images on their own)
- Text: Cuneiform in Unicode

Additionally, we provide the transliteration data.

- Text: Cuneiform Transliteration (downloaded from CDLI)

the `CuneiML_V1.1.json` is a list of dict as below:

```
{
    id: 105835, # CDLI ID
    img_url: , # CDLI img url
    lineart_url: , #  CDLI img url
    bboxes: [[181, 200, 601, 200], [..]], # list of bounding box of cutouts
    transliteration: {"obs": 
        ["4(disz) gu4 niga", "u4 2(u) 3(disz)-kam"],
         "res": [..], "left": [..]}, # 
    unicode: {"obs": 
        ["𒐉<S>𒄞<S>𒊺", "𒌓<S>𒌋𒌋<S>𒐈<S>𒄰"],
         "res": [..], "left": [..]}
}
```

### Getting the cutouts

1. Downlad the images on your own the photographs of each table from CDLI. `P100001.jpg` is the photograph of Tablet `id=100001`.

2. run the python code `python get_cutouts.py --bbox path_to_bbox_json --img path_to_img_folder`


### Getting Unicode for other transliteration

We also provide a script to convert any transliteration into Unicode.

run `python tokenize.py --input "4(disz) gu4 niga"` and you can get the unicode given the transliteration string.

Or, `python tokenize.py --input_file "PATH TO TEXT FILE" > unicode_output.txt` to convert the transliteration to unicode.

We also provide a `CuneiML_unicode.json` file to directly access the Cuneiform Unicode.

### Code

`get_cutouts`: the code to get major face cutouts. We only release the code and bounding boxes.

`cuneiform_unicode`: the code to convert transliteration (in ATF) to cuneiform unicode.