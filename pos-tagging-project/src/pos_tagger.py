import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn_crfsuite.metrics import flat_f1_score
import pickle

class POSTagger:
    def __init__(self):
        self.model = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        self.is_trained = False
    
    def read_conll_file(self, file_path):
        """
        .conll dosyasını okur ve cümleleri parse eder
        """
        sentences = []
        current_sentence = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                
                # Boş satır = cümle sonu
                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                else:
                    # Kelime ve etiket ayrımı
                    parts = line.split()
                    if len(parts) >= 2:
                        word = parts[0]
                        tag = parts[1]
                        current_sentence.append((word, tag))
            
            # Son cümleyi de ekle
            if current_sentence:
                sentences.append(current_sentence)
        
        return sentences
    
    def extract_features(self, sentence, i):
        """
        Bir kelimenin özelliklerini çıkarır
        """
        word = sentence[i][0]
        
        features = {
            'word.lower()': word.lower(),
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'word.length': len(word),
            'word.ispunct()': word in '.,;:!?-()[]{}"\'/\\',
            'word.isalpha()': word.isalpha(),
            'word.isalnum()': word.isalnum(),
        }
        
        # Kelime sonekleri (Türkçe için önemli) - sadece harf içeren kelimeler için
        if word.isalpha():
            if len(word) > 1:
                features['suffix_1'] = word[-1]
            if len(word) > 2:
                features['suffix_2'] = word[-2:]
            if len(word) > 3:
                features['suffix_3'] = word[-3:]
            
            # Kelime önekleri
            if len(word) > 1:
                features['prefix_1'] = word[0]
            if len(word) > 2:
                features['prefix_2'] = word[:2]
            if len(word) > 3:
                features['prefix_3'] = word[:3]
            
            # Türkçe'ye özel ek kontrolleri
            features['has_turkish_suffix'] = any(word.endswith(suffix) for suffix in 
                ['lar', 'ler', 'dan', 'den', 'ta', 'te', 'da', 'de', 'li', 'lı', 'lu', 'lü',
                 'sız', 'siz', 'suz', 'süz', 'mak', 'mek', 'yor', 'ken', 'ın', 'in', 'un', 'ün'])
        
        # Önceki kelime özellikleri
        if i > 0:
            prev_word = sentence[i-1][0]
            features['prev_word.lower()'] = prev_word.lower()
            features['prev_word.istitle()'] = prev_word.istitle()
            features['prev_word.ispunct()'] = prev_word in '.,;:!?-()[]{}"\'/\\'
            features['prev_word.isalpha()'] = prev_word.isalpha()
        else:
            features['BOS'] = True  # Beginning of sentence
        
        # Sonraki kelime özellikleri
        if i < len(sentence) - 1:
            next_word = sentence[i+1][0]
            features['next_word.lower()'] = next_word.lower()
            features['next_word.istitle()'] = next_word.istitle()
            features['next_word.ispunct()'] = next_word in '.,;:!?-()[]{}"\'/\\'
            features['next_word.isalpha()'] = next_word.isalpha()
        else:
            features['EOS'] = True  # End of sentence
        
        return features
    
    def sentence_to_features(self, sentence):
        """
        Bir cümlenin tüm kelimelerinin özelliklerini çıkarır
        """
        return [self.extract_features(sentence, i) for i in range(len(sentence))]
    
    def sentence_to_labels(self, sentence):
        """
        Bir cümlenin etiketlerini döndürür
        """
        return [label for word, label in sentence]
    
    def prepare_data(self, sentences):
        """
        Eğitim verilerini hazırlar
        """
        X = [self.sentence_to_features(sentence) for sentence in sentences]
        y = [self.sentence_to_labels(sentence) for sentence in sentences]
        return X, y
    
    def train(self, file_path, test_size=0.2):
        """
        Modeli eğitir
        """
        print("Veri okunuyor...")
        sentences = self.read_conll_file(file_path)
        print(f"Toplam {len(sentences)} cümle okundu.")
        
        print("Özellikler çıkarılıyor...")
        X, y = self.prepare_data(sentences)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Eğitim: {len(X_train)} cümle, Test: {len(X_test)} cümle")
        print("Model eğitiliyor...")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Test performansı
        y_pred = self.model.predict(X_test)
        
        print("\n=== Test Sonuçları ===")
        print(f"F1 Score: {flat_f1_score(y_test, y_pred, average='weighted'):.4f}")
        print("\nDetaylı Rapor:")
        print(flat_classification_report(y_test, y_pred))
        
        return X_test, y_test, y_pred
    
    def tokenize_sentence(self, sentence_text):
        """
        Cümleyi kelimelere ve noktalama işaretlerine ayırır
        """
        # Noktalama işaretlerini ayır
        # Noktalama işaretleri öncesi ve sonrası boşluk koy
        sentence_text = re.sub(r'([.!?,:;(){}[\]"\'-])', r' \1 ', sentence_text)
        
        # Fazla boşlukları temizle
        sentence_text = re.sub(r'\s+', ' ', sentence_text).strip()
        
        # Kelimelere ayır ve boş olanları filtrele
        words = [word for word in sentence_text.split() if word.strip()]
        
        return words
    
    def predict_sentence(self, sentence_text):
        """
        Yeni bir cümleyi etiketler
        """
        if not self.is_trained:
            raise Exception("Model henüz eğitilmedi! Önce train() metodunu çağırın.")
        
        # Cümleyi doğru şekilde tokenize et
        words = self.tokenize_sentence(sentence_text)
        
        # Sahte sentence formatı oluştur (etiket kısmı boş)
        sentence = [(word, '') for word in words]
        
        # Özellikleri çıkar
        features = self.sentence_to_features(sentence)
        
        # Tahmin yap
        predicted_tags = self.model.predict([features])[0]
        
        # Sonuçları döndür
        result = []
        for word, tag in zip(words, predicted_tags):
            result.append((word, tag))
        
        return result
    
    def save_model(self, file_path):
        """
        Eğitilmiş modeli kaydet
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model {file_path} dosyasına kaydedildi.")
    
    def load_model(self, file_path):
        """
        Kaydedilmiş modeli yükle
        """
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"Model {file_path} dosyasından yüklendi.")

# Kullanım örneği
if __name__ == "__main__":
    # POS Tagger oluştur
    tagger = POSTagger()
    
    # Modeli eğit (dosya yolunu kendi dosyanızla değiştirin)
    tagger.train('dataset_pos_tagget2.conll')
    
    # Modeli kaydet
    tagger.save_model('pos_model.pkl')
    
    # Test cümlesi
    test_sentence = "Bugün canım istememesine rağmen ders çalıştım."
    result = tagger.predict_sentence(test_sentence)
    
    print(f"\n=== '{test_sentence}' Cümlesi İçin Tahminler ===")
    for word, tag in result:
        print(f"{word} -> {tag}")