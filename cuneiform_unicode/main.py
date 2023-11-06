import re
import json
import pickle as pkl


def get_token_mapping(token_path="./token.tsv", token_path2="./token.tsv"):
    text2sign = {}
    sign2text = {}
    for t in open(token_path).readlines():
        try:
            k, s = t.strip("\n").split("\t")
        except:
            print(t)
        sign2text[s] = k
        text2sign[k] = s
        
    for t in open(token_path2).readlines():
        try:
            k, s = t.strip("\n").split("\t")
        except:
            print(t)
        sign2text[s] = k
        text2sign[k] = s
        
    return text2sign, sign2text

text2sign, sign2text = get_token_mapping("./cuneiform_vocab.txt", "./token.tsv")


# speical token

s_tokens = ['<B>', # borken
            '<M>', # missing one or more token?
            "<S>", # blank space
            "<D>", # divine
            "<munus>", # young woman, or woman
            "<ansze>",
            "<ki>",
            "<disz>",
            "x", # uknown signs
            ]
from collections import Counter
vocab_freq = Counter()
new_tokens = Counter()
langs = Counter()
unknown_faces = Counter()

def remove_at(x):
    if x.endswith("@c)") or x.endswith("@t)"):
        return x[:-3] + ")"



def tokenize(raw_text, info={}):
    token_text = {"default": []}
    curr_face = "default"
    for line in raw_text.split("\n"):
        line = line.strip()
        # print(line)
        if line.startswith("&") or line.startswith("'&"):
            pass
            # the metadata
        elif line.startswith("#atf"):
            info['lang'] = line.split("lang ")[-1].strip()
            langs[info['lang']] += 1
            if info['lang'] in ['sux', 'akk', 'sux, akk']:
                continue
            else:
                # do not process those not sum or akk
                return info
        elif line.startswith("#") or line.startswith(">>"):
            # comment
            continue
        elif line.startswith("$"):
            if 'broken' in line:
                try:
                    token_text[curr_face].append("<B>")
                except:
                    continue
        elif line.startswith('@'):
            key = line[1:].strip().strip("?")
            if key in ['obverse', 'reverse', 'left', 'right', 'top', 'down', 
                       "surface a"]:
                curr_face = key
                token_text[key] = []
            # elif key.startswith("column"):
            # elif key
            elif key.startswith("column"):
                token_text[curr_face].append("<COL>")
            else:
                unknown_faces[key] += 1
            
        else:
            # speicial symbols
            line = line.replace("{d}", "<D>")
            
            for x in re.findall("\{.*?\}", line):
                line = line.replace(x, " " + x[1:-1] + " ")
            
            # line = line.replace("{munus}", " <munus> ")
            # line = line.replace("{ansze}", " <ansze> ")
            # line = line.replace("{ki}", " <ki> ")
            # line = line.replace("{disz}", " <disz> ")
            line = line.replace("($ blank space $)", "<S>")
            
            # remove underscore
            line = line.replace("_", " ")
            
            # remove ending hash #
            line = line.replace("#", "")
            
            # remove question mark, exclamation mark
            line = line.replace("?", "")
            line = line.replace("!", "")
            
            # remove [] and ()
            for x in re.findall("\[.*?\]", line):
                line = line.replace(x, "")
            # print("\t\t>>>", line)
            line = line.split(". ")
            
            if len(line) >= 2:
                # make sure only leading line number is split
                if len(line) > 2:
                    line = line[0], ". ".join(line[1:])
                    
                line_num, text = line
                if curr_face != "":
                    tokens = text.split(" ")
                    signs = []
                    for i, t in enumerate(tokens):
                        if i > 0 and len(signs) > 0:
                            signs.append("<S>") # insert a space between words
                            
                        if "-" in t:
                            ts = t.split("-")
                            for x in ts:
                                x = x.strip()
                                if len(x) == 0:
                                    continue
                                if x in text2sign:
                                    vocab_freq[x] += 1
                                    signs.append(text2sign[x])
                                else:
                                    new_x = remove_at(x)
                                    if new_x in text2sign:
                                        signs.append(text2sign[new_x])
                                    else:
                                        new_tokens[x] += 1
                                    # print(x)
                        elif t in text2sign:
                            signs.append(text2sign[t])
                        elif t in s_tokens:
                            vocab_freq[t] += 1
                            signs.append(t)
                        else:
                            new_x = remove_at(t)
                            if new_x in text2sign:
                                signs.append(text2sign[new_x])
                            else:
                                if len(t.strip()) > 0:
                                    new_tokens[t] += 1
                                # print(t)
                    token_text[curr_face].append({'raw': text, 
                                                  'num': line_num, 
                                                  "sign": signs
                                                  })
    info['text'] = token_text
    # print(info)
    # print(token_text)
    return info


def get_text(data, idx):
    # the pkl file store the text at idx 44
    return data[idx][44]


if __name__ == "__main__":

    data = pkl.load(open("/trunk2/datasets/cuneiform/raw_data.pkl", 'rb'))
    image_anno = json.load(open("/trunk2/datasets/cuneiform/image_anno.json"))
    
    new_image_anno = {}

    for idx in image_anno:
        if 'text' in image_anno[idx]:
            raw_text = get_text(data, int(idx))
            new_image_anno[int(idx)] = tokenize(raw_text)





