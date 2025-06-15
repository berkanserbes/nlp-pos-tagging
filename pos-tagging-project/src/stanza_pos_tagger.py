import stanza

#nlp = stanza.Pipeline(lang='tr', processors='tokenize,pos')

nlp = stanza.Pipeline(lang='tr', processors='tokenize,pos')

input_file = "dataset.txt"
output_file = "dataset_pos_tagged.conll"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        sentence = line.strip()
        if not sentence:
            continue  # Boş satır varsa atla
        
        # --- 4. Cümleyi Stanza'ya verip işlem yaptırıyoruz ---
        doc = nlp(sentence)
        
        # --- 5. Her kelimeyi ve POS etiketini yazdırıyoruz ---
        for sent in doc.sentences:
            for word in sent.words:
                # word.text: kelime
                # word.upos: Universal POS etiketi (NOUN, VERB, ADJ, vb.)
                f_out.write(f"{word.text}\t{word.upos}\n")
            f_out.write("\n")  # Cümle sonu için boş satır (CoNLL formatında önemli)

print(f"POS etiketleme tamamlandi: {output_file}")