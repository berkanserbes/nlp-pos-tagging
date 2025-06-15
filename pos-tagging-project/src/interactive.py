from pos_tagger import POSTagger

def main():
    print("POS Tagging Sistemi")
    print("=" * 50)
    
    tagger = POSTagger()
    
    while True:
        print("\nSeçenekler:")
        print("1. Yeni model eğit")
        print("2. Kaydedilmiş model yükle")
        print("3. Cümle etiketle")
        print("4. Çıkış")
        
        choice = input("\nSeçiminiz (1-4): ").strip()
        
        if choice == "1":
            train_new_model(tagger)
        
        elif choice == "2":
            load_existing_model(tagger)
        
        elif choice == "3":
            if not tagger.is_trained:
                print("Önce bir model eğitmeniz veya yüklemeniz gerekiyor!")
                continue
            
            tag_sentence(tagger)
        
        elif choice == "4":
            print("Görüşürüz!")
            break
        
        else:
            print("Geçersiz seçim!")

def train_new_model(tagger):
    print("\nYeni Model Eğitimi")
    print("-" * 30)
    
    file_path = input("CONLL dosya yolu: ").strip()
    
    try:
        # Test oranını sor
        test_size = input("Test oranı (0.2): ").strip()
        if not test_size:
            test_size = 0.2
        else:
            test_size = float(test_size)
        
        # Modeli eğit
        tagger.train(file_path, test_size=test_size)
        
        # Kayıt seçeneği
        save_choice = input("\nModeli kaydetmek ister misiniz? (e/h): ").strip().lower()
        if save_choice in ['e', 'evet', 'y', 'yes']:
            model_name = input("Model dosya adı (pos_model.pkl): ").strip()
            if not model_name:
                model_name = "pos_model.pkl"
            
            tagger.save_model(model_name)
        
        print("Model eğitimi tamamlandı!")
        
    except Exception as e:
        print(f"Hata: {e}")

def load_existing_model(tagger):
    print("\n Model Yükleme")
    print("-" * 20)
    
    file_path = input("Model dosya yolu: ").strip()
    
    try:
        tagger.load_model(file_path)
        print("Model başarıyla yüklendi!")
        
    except Exception as e:
        print(f"Hata: {e}")

def tag_sentence(tagger):
    print("\nCümle Etiketleme")
    print("-" * 25)
    
    while True:
        sentence = input("\nCümle girin (çıkmak için 'q'): ").strip()
        
        if sentence.lower() == 'q':
            break
        
        if not sentence:
            print("Boş cümle girmeyin!")
            continue
        
        try:
            result = tagger.predict_sentence(sentence)
            
            print(f"\n'{sentence}' - Etiketleme Sonucu:")
            print("-" * 60)
            
            for word, tag in result:
                print(f"{word:15} -> {tag}")
            
            # İstatistikler
            tag_counts = {}
            for _, tag in result:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            print(f"\nEtiket Dağılımı:")
            for tag, count in sorted(tag_counts.items()):
                print(f"  {tag}: {count} kelime")
                
        except Exception as e:
            print(f"Hata: {e}")

def demo_with_sample_sentences():
    """
    Örnek cümlelerle demo yapar
    """
    sample_sentences = [
        "Bugün canım istememesine rağmen ders çalıştım.",
        "Yeni bir dil öğrenmek farklı bir kültürün kapılarını aralamaktır.",
        "Kodun bu satırında asenkron işlem tamamlanmadan sonraki adıma geçiliyor olabilir.",
        "Güzel bir hava var bugün.",
        "Projem için veri toplamaya başladım."
    ]
    
    print("\nDemo Modunda Çalışıyorum...")
    print("Örnek cümleler etiketleniyor:\n")
    
    tagger = POSTagger()
    
    try:
        tagger.load_model('pos_model.pkl')
        
        for sentence in sample_sentences:
            result = tagger.predict_sentence(sentence)
            
            print(f"'{sentence}'")
            print("-" * (len(sentence) + 10))
            
            for word, tag in result:
                print(f"  {word:15} -> {tag}")
            print()
    
    except Exception as e:
        print(f"Demo çalıştırılamadı: {e}")
        print("Önce bir model eğitmeniz veya yüklemeniz gerekiyor.")

if __name__ == "__main__":
    main()