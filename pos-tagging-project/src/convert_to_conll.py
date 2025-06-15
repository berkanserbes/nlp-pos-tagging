import stanza
import re

def preprocess_text(text: str) -> str:
    """Metni temizle ve hazırla"""
    text = re.sub(r'[^\w\s]', '', text)  # Özel karakterleri temizle
    text = text.lower()
    return text

def main():
    nlp = stanza.Pipeline(lang='tr', processors='tokenize,pos')
    
    with open("dataset.txt", "r", encoding="utf-8") as f_in, \
         open("dataset_pos_tagged.conll", "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            sentence = line.strip()
            if not sentence:
                continue
            
            # Cümleyi temizle
            clean_sentence = preprocess_text(sentence)
            
            doc = nlp(clean_sentence)
            
            # Her kelime için POS etiketini yaz
            for sent in doc.sentences:
                for word in sent.words:
                    f_out.write(f"{word.text}\t{word.upos}\n")
            f_out.write("\n")  # Cümle sonu için boş satır

if __name__ == "__main__":
    main()
