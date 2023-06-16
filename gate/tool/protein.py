import os, sys, argparse

def get_sequence(contents):
    """Enclosing logic in a function to simplify code"""

    res_codes = [
        # 20 canonical amino acids
        ('CYS', 'C'), ('ASP', 'D'), ('SER', 'S'), ('GLN', 'Q'),
        ('LYS', 'K'), ('ILE', 'I'), ('PRO', 'P'), ('THR', 'T'),
        ('PHE', 'F'), ('ASN', 'N'), ('GLY', 'G'), ('HIS', 'H'),
        ('LEU', 'L'), ('ARG', 'R'), ('TRP', 'W'), ('ALA', 'A'),
        ('VAL', 'V'), ('GLU', 'E'), ('TYR', 'Y'), ('MET', 'M'),
        # Non-canonical amino acids
        # ('MSE', 'M'), ('SOC', 'C'),
        # Canonical xNA
        ('  U', 'U'), ('  A', 'A'), ('  G', 'G'), ('  C', 'C'),
        ('  T', 'T'),
    ]

    three_to_one = dict(res_codes)
    # _records = set(['ATOM  ', 'HETATM'])
    _records = set(['ATOM  '])

    sequence = []
    read = set()
    for line in contents:
        line = line.strip()
        if line[0:6] in _records:
            resn = line[17:20]
            resi = line[22:26]
            icode = line[26]
            r_uid = (resn, resi, icode)
            if r_uid not in read:
                read.add(r_uid)
            else:
                continue
            aa_resn = three_to_one.get(resn, 'X')
            sequence.append(aa_resn)

    return ''.join(sequence)

    
def split_pdb(complex_pdb: str, outdir: str):
    chain_models = []
    pre_chain = None
    i = 0
    for line in open(complex_pdb, 'r').readlines():
        if not line.startswith('ATOM'):
            continue
        chain_name = line[21]
        if pre_chain is None:
            pre_chain = chain_name
            chain_models += [outdir + '/' + chain_name + '.pdb']
            fw = open(outdir + '/' + chain_name + '.pdb', 'w')
            fw.write(line[:21] + ' ' + line[22:])
        elif chain_name == pre_chain:
            fw.write(line[:21] + ' ' + line[22:])
        else:
            fw.close()
            i = i + 1
            chain_models += [outdir + '/' + chain_name + '.pdb']
            fw = open(outdir + '/' + chain_name + '.pdb', 'w')
            fw.write(line[:21] + ' ' + line[22:])
            pre_chain = chain_name
    fw.close()
    return chain_models
