import os, sys, argparse
from Bio.PDB.PDBParser import PDBParser
import numpy as np

def get_sequence(inpdb):
    """Enclosing logic in a function to simplify code"""

    seq_to_res_mapping = []
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
    for line in open(inpdb):
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
            seq_to_res_mapping += [int(resi)]

    return {'sequence': ''.join(sequence), 'mapping': seq_to_res_mapping}

    
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


def pad_line(line):
    """Helper function to pad line to 80 characters in case it is shorter"""
    size_of_line = len(line)
    if size_of_line < 80:
        padding = 80 - size_of_line + 1
        line = line.strip('\n') + ' ' * padding + '\n'
    if '\n' not in line[:81]:
        return line
    return line[:81]  # 80 + newline character


def write_pdb_file(new_pdb, pdb_file):
    if os.path.exists(pdb_file):
        os.remove(pdb_file)
    try:
        _buffer = []
        _buffer_size = 5000  # write N lines at a time
        for lineno, line in enumerate(new_pdb):
            if not (lineno % _buffer_size):
                open(pdb_file, 'a').write(''.join(_buffer))
                _buffer = []
            _buffer.append(line)

        open(pdb_file, 'a').write(''.join(_buffer))
    except IOError:
        # This is here to catch Broken Pipes
        # for example to use 'head' or 'tail' without
        # the error message showing up
        pass


#starting_resid = -1
def renumber_residues(fhandle, starting_resid):
    """Resets the residue number column to start from a specific number.
    """
    _pad_line = pad_line
    prev_resid = None  # tracks chain and resid
    resid = starting_resid - 1  # account for first residue
    records = ('ATOM', 'HETATM', 'TER', 'ANISOU')
    for line in fhandle:
        line = _pad_line(line)
        if line.startswith(records):
            line_resuid = line[17:27]
            if line_resuid != prev_resid:
                prev_resid = line_resuid
                resid += 1
                if resid > 9999:
                    emsg = 'Cannot set residue number above 9999.\n'
                    sys.stderr.write(emsg)
                    sys.exit(1)

            yield line[:22] + str(resid).rjust(4) + line[26:]

        else:
            yield line

#starting_resid = -1
def renumber_residues_with_order(fhandle, reorder_indices):
    """Resets the residue number column to start from a specific number.
    """
    _pad_line = pad_line
    prev_resid = None  # tracks chain and resid
    counter = -1
    records = ('ATOM', 'HETATM', 'TER', 'ANISOU')
    for line in fhandle:
        line = _pad_line(line)
        if line.startswith(records):
            line_resuid = line[17:27]
            if line_resuid != prev_resid:
                prev_resid = line_resuid
                counter += 1
                if reorder_indices[counter] > 9999:
                    emsg = 'Cannot set residue number above 9999.\n'
                    sys.stderr.write(emsg)
                    sys.exit(1)

            yield line[:22] + str(reorder_indices[counter]).rjust(4) + line[26:]

        else:
            yield line


#starting_value = -1
def renumber_atom_serials(fhandle, starting_value):
    """Resets the atom serial number column to start from a specific number.
    """

    # CONECT 1179  746 1184 1195 1203
    fmt_CONECT = "CONECT{:>5s}{:>5s}{:>5s}{:>5s}{:>5s}" + " " * 49 + "\n"
    char_ranges = (slice(6, 11), slice(11, 16),
                   slice(16, 21), slice(21, 26), slice(26, 31))

    serial_equiv = {'': ''}  # store for conect statements

    serial = starting_value
    records = ('ATOM', 'HETATM')
    for line in fhandle:
        if line.startswith(records):
            serial_equiv[line[6:11].strip()] = serial
            yield line[:6] + str(serial).rjust(5) + line[11:]
            serial += 1
            if serial > 99999:
                emsg = 'Cannot set atom serial number above 99999.\n'
                sys.stderr.write(emsg)
                sys.exit(1)

        elif line.startswith('ANISOU'):
            # Keep atom id as previous atom
            yield line[:6] + str(serial - 1).rjust(5) + line[11:]

        elif line.startswith('CONECT'):
            # 6:11, 11:16, 16:21, 21:26, 26:31
            serials = [line[cr].strip() for cr in char_ranges]

            # If not found, return default
            new_serials = [str(serial_equiv.get(s, s)) for s in serials]
            conect_line = fmt_CONECT.format(*new_serials)

            yield conect_line
            continue

        elif line.startswith('MODEL'):
            serial = starting_value
            yield line

        elif line.startswith('TER'):
            yield line[:6] + str(serial).rjust(5) + line[11:]
            serial += 1

        else:
            yield line


def keep_residues(fhandle, residue_range):
    """Deletes residues within a certain numbering range.
    """
    prev_res = None
    records = ('ATOM', 'HETATM', 'TER', 'ANISOU')
    for line in fhandle:
        if line.startswith(records):

            res_id = line[21:26]  # include chain ID
            if res_id != prev_res:
                prev_res = res_id

            if int(line[22:26]) not in residue_range:
                continue

        yield line


def reindex_pdb_file(in_file, out_file, keep_indices=None, reorder_indices=None):
    # print(keep_indices)
    fhandle = open(in_file, 'r')
    if keep_indices is not None:
        fhandle = keep_residues(fhandle, keep_indices)

    if reorder_indices is not None:
        fhandle = renumber_residues_with_order(fhandle, reorder_indices)
    else:
        fhandle = renumber_residues(fhandle, 1)

    fhandle = renumber_atom_serials(fhandle, 1)
    write_pdb_file(fhandle, out_file)


def cal_contact_number(pdb1, L1, pdb2, L2, distance_threshold=8):
    parser = PDBParser(QUIET=True)
    structure1 = parser.get_structure('', pdb1)
    structure2 = parser.get_structure('', pdb2)
    # print(pdb2)

    model1 = structure1[0]
    chain_id1 = list(model1.child_dict.keys())
    xyzPDB1 = model1[chain_id1[0]]

    model2 = structure2[0]
    chain_id2 = list(model2.child_dict.keys())
    xyzPDB2 = model2[chain_id2[0]]

    h_dist_map = np.full((L1, L2), -1)
    # 1: contact, 0: not contact
    h_contact_map = np.zeros((L1, L2))
    for i, res_i in enumerate(xyzPDB1.get_residues()):
        for j, res_j in enumerate(xyzPDB2.get_residues()):
            ca_distance = -1
            for atom_i in res_i:
                if atom_i.name != 'CA':
                    continue
                for atom_j in res_j:
                    if atom_j.name != 'CA':
                        continue
                    ca_distance = atom_i - atom_j
            h_dist_map[i - 1, j - 1] = ca_distance
            if ca_distance >= 0 and ca_distance < distance_threshold:
                h_contact_map[i - 1, j - 1] = 1
    
    return (h_contact_map > 0).sum(), h_contact_map, h_dist_map

def cal_contact_number_heavy(pdb1, L1, pdb2, L2, distance_threshold=5):
    parser = PDBParser(QUIET=True)
    structure1 = parser.get_structure('', pdb1)
    structure2 = parser.get_structure('', pdb2)
    # print(pdb2)

    model1 = structure1[0]
    chain_id1 = list(model1.child_dict.keys())
    xyzPDB1 = model1[chain_id1[0]]

    model2 = structure2[0]
    chain_id2 = list(model2.child_dict.keys())
    xyzPDB2 = model2[chain_id2[0]]

    h_dist_map = np.full((L1, L2), -1)
    # 1: contact, 0: not contact
    h_contact_map = np.zeros((L1, L2))
    for i, res_i in enumerate(xyzPDB1.get_residues()):
        for j, res_j in enumerate(xyzPDB2.get_residues()):
            dist_list = []
            for atom_i in res_i:
                for atom_j in res_j:
                    if ('C' in atom_i.name or 'N' in atom_i.name or 'O' in atom_i.name or 'S' in atom_i.name) and \
                        ('C' in atom_j.name or 'N' in atom_j.name or 'O' in atom_j.name or 'S' in atom_j.name):
                        dist_list.append(atom_i - atom_j)
                    else:
                        continue
            min_dist = np.min(dist_list)          

            h_dist_map[i - 1, j - 1] = min_dist

            if min_dist >= 0 and min_dist < distance_threshold:

                h_contact_map[i - 1, j - 1] = 1
    
    return (h_contact_map > 0).sum(), h_contact_map, h_dist_map


def read_files_by_prefix_and_ext(indir, prefix='', ext='', full_path=True):
    paths = []
    for name in os.listdir(indir):
        if len(prefix) > 0 and name.find(prefix) != 0:
            continue
        
        if len(ext) > 0 and name.split('.')[-1] != ext:
            continue

        if full_path:
            paths += [indir + '/' + name]
        else:
            paths += [name]
    return paths