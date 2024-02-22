# CuneiML

This repository hosts the code to get the artifects of Cuneiform in the paper [CuneiML: A Cuneiform Dataset for Machine Learning](https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.151). 


```
@article{Chen-2023,
 author = {Chen, Danlu and Agarwal, Aditi and Berg-Kirkpatrick, Taylor and Myerston, Jacobo},
 doi = {10.5334/johd.151},
 journal = {Journal of Open Humanities Data},
 month = {Dec},
 title = {CuneiML: A Cuneiform Dataset for Machine Learning},
 year = {2023}
}
```

### Data

 We provoide the dehygration version of the data in `CuneiML_V1.2.json`:

- **Cutouts (image)**: major face cutouts (we provide the bounding boxes only, and users need to obtain the orignal images on their own because of copyright)
- **Unicode (text)**: Cuneiform in Unicode

Additionally, we provide the transliteration data we obtained from CDLI.

- **Transliteration (text)**: Cuneiform Transliteration (downloaded from CDLI)


The file **iid_split.json** provides the CDLI ID of train/valid/test split for the time period classification experiment.


---
the `CuneiML_V1.2.json` is a list of dict as below:

```
{
 'id': 131837,
 'img_url': 'https://cdli.mpiwg-berlin.mpg.de/dl/photo/P131837.jpg',    # link to photo
 'lineart': 'https://cdli.mpiwg-berlin.mpg.de/dl/lineart/P131837_l.jpg',# link to lineart
 'bboxes': ((204.0, 200.0), (523.0, 522.0)),                               # the bounding box
 'text': {
   'obverse': [
      {'raw': '2(gesz2) 4(asz) 2(barig) 4(disz) sila3 gur',
         'num': '1',
         'sign': ['𒐂', '𒐉', '𒋡', '𒄥']},
      {'raw': 'a2 lu2 hun-ga2', 
         'num': '2', 
         'sign': ['𒀉', '𒇽', '𒂠', '𒂷']},
      {'raw': 'ugu2 lu2- <D> inanna ba-a-gar',
         'num': '3',
         'sign': ['𒀀𒅗', '𒇽', '<D>', '𒈹', '𒁀', '𒀀', '𒃻']}],
   'reverse': [
      {'raw': 'mu sza-asz-ru ki  ba-hul',
       'num': '1',
       'sign': ['𒈬', '𒊭', '𒀸', '𒊒', '𒆠', '𒁀', '𒅆𒌨']}]
   }
}
```

The `raw` field is the transliteration obtained from CDLI and the `sign` field is the Cuneiform Unicode of the lines. The `num` field is from CDLI's line label.

Note that around 1% of the cuneiform Unicode is not convert automatically.

### Getting the cutouts

1. Downlad the images on your own the photographs of each table from CDLI. `P100001.jpg` is the photograph of Tablet `id=100001`. For example:

```python
import requests

def download_image(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Image downloaded successfully: {filename}")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")

download_image(url="https://cdli.mpiwg-berlin.mpg.de/dl/photo/P131837.jpg", filename="P131837.jpg")
```

2. Cut the image using the PIL package:
   ```python
    from PIL import Image
    im = Image.open("P131837.jpg")
    im.crop((204.0, 200.0, 523.0, 522.0)).save("P131837_cut.jpg")
   ```


### Code

We also release the code to get the cutouts/unicode for new data that not includes in the current selection of dataset.

`get_cutouts`: the code to get major face cutouts.  Please refer to `get_cutouts/README.md` for details.

`cuneiform_unicode`: the code to convert transliteration (in ATF) to cuneiform unicode.

#### Getting cutouts

(WIP) 

#### Getting Unicode for other transliteration

We also provide a script to convert any transliteration into Unicode.

run `cd cuneiform_unicode; python main.py --raw_text "1. 4(disz) gu4 niga \n 2. _mu us2-sa {kusz}a2-la2 e2 {d}nanna-ra a mu-na-ru"` and you can get the unicode given the transliteration string.

Note that the code primary design for parsing CDLI's ATF and therefore a leading line number is reqiured for the `raw_text`. Multiple lines are seperated by `\n`.

### Change logs

2024.02.21 Update image urls, more instruction