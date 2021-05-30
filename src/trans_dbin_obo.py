# Date: 2021/4/2
# Author: Qianqian Peng
import nltk
import argparse

def dice_coefficient_score(s1:str,s2:str):
    set1 = set(s1)
    set2 = set(s2)
    inter_set = set1.intersection(set2)
    return 2*len(inter_set)/(len(set1)+len(set2))

def main(args):
    head = 'format-version: 1.0\ndata-version: MeSH/releases/2021-04-01\n' \
           'saved-by: Qianqian Peng\nontology: MeSH: d2020.bin'
    add_synonym = args.addsyn
    add_all_syn = True
    dbin_path = args.infile
    if not add_synonym:
        out_path = args.outfile
    else:
        out_path = '.'.join(args.outfile.split('.')[0:-1]) + '_addsyn' + '.obo'  # '../Data/MeSH/mesh_addsyn.obo'
    if add_all_syn:
        out2 = open('../ontology/all_name_id.txt', 'w', encoding='utf-8', newline='\n')
    data = open(dbin_path,'r',encoding='utf-8').read()
    out = open(out_path,'w',encoding='utf-8',newline='\n')
    out.write(head + '\n\n')
    data = data.split('\n\n')
    data.pop(-1)
    data_num = len(data)
    data_obo = []

    #set the root node
    out.write('[Term]'+'\n')
    out.write('id: '+'MS:D000000' + '\n')
    out.write('name: '+'All'+'\n')
    out.write('\n')
    n = 0
    for _ in range(data_num):
        n += 1
        print(n,'/',data_num)
        one_record = data.pop(0)
        one_record = one_record.split('\n')
        new_record = {}
        synonyms = []
        synonyms2 = []
        for elem in one_record:
            if elem.startswith('MH = '):
                new_record['name'] = elem.strip()[len('MH = '):]
            if elem.startswith('UI = '):
                new_record['id'] = 'MS:' + elem.strip()[len('UI = '):]
            if add_synonym:
                if elem.startswith('ENTRY = ') or elem.startswith('PRINT ENTRY = '):
                    elem = elem.strip().split(' = ')[1].split('|')[0]
                    edit_dist = nltk.edit_distance(elem.lower(),new_record['name'].lower())
                    dice_score = dice_coefficient_score(elem.lower(),new_record['name'].lower())
                    synonyms2.append(elem)
                    if edit_dist/len(elem) <= 0.25 or edit_dist/len(new_record['name']) <= 0.25 or dice_score >= 0.75:
                        synonyms.append(elem)
        new_record['is_a'] = 'MS:D000000'
        out.write('[Term]'+'\n')
        out.write('id: '+new_record['id'] + '\n')
        out.write('name: '+new_record['name']+'\n')
        out.write('is_a: '+new_record['is_a']+'\n')
        if add_all_syn:
            out2.write(new_record['name']+'::'+new_record['id']+'\n')
            for s in synonyms2:
                out2.write(s+'::'+new_record['id']+'\n')
        if add_synonym:
            if synonyms != []:
                synonyms = list(set(synonyms))
                synonyms = synonyms[0:3 if len(synonyms) >= 3 else len(synonyms)]
                for syn in synonyms:
                    out.write("synonym: " + '"' + syn + '"' + ' EXACT []'+'\n')
        out.write('\n')
    out.close()
    print('done...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='convert MeSH d2020.bin to mesh.obo.')
    parser.add_argument('--infile', '-i', help="input file path", default='../ontology/d2020.bin')
    parser.add_argument('--outfile', '-o', help="output file path", default='../ontology/mesh.obo')
    parser.add_argument('--addsyn', '-a', help="output file path", default=True)
    args = parser.parse_args()
    main(args)
