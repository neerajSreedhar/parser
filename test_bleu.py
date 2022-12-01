from nltk.translate.bleu_score import sentence_bleu

# reference = '<SOS> place banana on camera <SEP> place orange on <EOS>'.split()
# candidate = '<SOS> place banana on camera <SEP> place orange on camera <EOS>'.split()
# # print('BLEU score -> {}'.format(sentence_bleu([reference], candidate, weights=(1,0,0,0))))
# print('BLEU score -> {}'.format(sentence_bleu([reference], candidate)))

with open('pred/vanilla_clip/99.txt', 'r') as f:
    bleus = []
    buffer_sentences = [''] * 2
    for idx, line in enumerate(f):
        # print(idx, line)
        if idx % 3 == 0:
            buffer_sentences[0] = line.strip()
        elif idx % 3 == 1:
            buffer_sentences[1] = '<SOS> ' + line.strip()
        else:
            reference = buffer_sentences[0].split()
            candidate = buffer_sentences[1].split()
            bleu = sentence_bleu([reference], candidate)
            bleus.append(bleu)
            print(idx, buffer_sentences, reference, candidate, bleu)

print(sum(bleus) / len(bleus))