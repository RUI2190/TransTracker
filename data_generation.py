import random
import copy

def gen_tra(l):
    half = l // 3
    seq1 = gen_dna_sequence(half)
    seq2 = gen_dna_sequence(l - 2 * half)
    return seq1 + seq2 + rev_comp(seq1)

def gen_tsd(l):
    return gen_dna_sequence(l)

def mutate_tsd(dna, p):
    if random.random() < p:
        pos = random.randint(0, len(dna) - 1)
        nucleotides = ['A', 'C', 'G', 'T']
        current_nucleotide = dna[pos]
        nucleotides.remove(current_nucleotide)
        new_nucleotide = random.choice(nucleotides)
        dna = dna[:pos] + new_nucleotide + dna[pos + 1:]

    return dna

def rev_comp(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[nuc] for nuc in reversed(seq))

def gen_dna_sequence(length):
    return ''.join(random.choice('ACGT') for _ in range(length))

def gen_dna_population(n, length_range):
    return [gen_dna_sequence(random.randint(*length_range)) for _ in range(n)]

def simulate_tra_movement(n, dna_length_range, tra_length_range, tsd_length_range, p): # p --> probability of hamming distance 1 change in tsd
    population = gen_dna_population(n, dna_length_range)
    original_population = copy.deepcopy(population)
    tsd_length = random.randint(*tsd_length_range)
    tsd = gen_tsd(tsd_length)
    orig_tsd = tsd
    tra_length = random.randint(*tra_length_range)
    tra = gen_tra(tra_length)
    insertion_details = []
    insertion_path = []

    # first tsd exactly original tsd
    insertion_idx = random.randint(0, len(population[0]) - 1)
    insertion_path.append((tsd, 0, insertion_idx))
    insertion_details.append(insertion_idx)
    population[0] = population[0][:insertion_idx] + tsd + tsd + population[0][insertion_idx:]

    # previously visited dna seq
    for i in range(1, n - 1):
      tsd = mutate_tsd(tsd, p)
      insertion_idx = random.randint(0, len(population[i]) - 1)
      insertion_path.append((tsd, i, insertion_idx))
      insertion_details.append(insertion_idx)
      population[i] = population[i][:insertion_idx] + tsd + tsd + population[i][insertion_idx:]
    # where the tra resides
    tsd = mutate_tsd(tsd, p)
    insertion_idx = random.randint(0, len(population[n - 1]) - 1)
    insertion_path.append((tsd, n - 1, insertion_idx))
    insertion_details.append(insertion_idx)
    population[n - 1] = population[i][:insertion_idx] + tsd + tra + tsd + population[i][insertion_idx:]

    return orig_tsd, tra, original_population, population, insertion_path, insertion_details

def print_orig(population):
  for idx, dna in enumerate(population):
    print("DNA " + str(idx + 1) + ": " + dna)

def print_with_markers(population, tsd, tra, insertion_details):
  for i in range(len(population) - 1):
    population[i] = population[i][:insertion_details[i]] + "~" + population[i][insertion_details[i]:insertion_details[i] + 2 * len(tsd)] + "~" + population[i][insertion_details[i] + 2 * len(tsd):]
  population[-1] = population[-1][:insertion_details[-1]] + "~" + population[-1][insertion_details[-1]:insertion_details[-1] + 2 * len(tsd) + len(tra)] + "~" + population[-1][insertion_details[-1] + 2 * len(tsd) + len(tra):]
  for idx, dna in enumerate(population):
    print("DNA " + str(idx + 1) + ": " + dna)