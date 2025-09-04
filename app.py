"""
Pusula Veri Bilimi Staj Case Study'si

Yazar: Mert güleç
E-posta: mertgg54@gmail.com
Tarih: Eylül 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as mat
import seaborn as sea
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
import os
warnings.filterwarnings('ignore')

mat.style.use('default')
sea.set_palette("husl")
mat.rcParams['font.size'] = 10
mat.rcParams['axes.titlesize'] = 12

class VeriAnalizci:
    def __init__(self, veri_yolu):
        self.veri_yolu = veri_yolu
        self.df = None
        self.df_islenmis = None
        
    def veri_yukle(self):
        try:
            dosya_uzantisi = os.path.splitext(self.veri_yolu)[1].lower()
            
            if dosya_uzantisi == '.xlsx' or dosya_uzantisi == '.xls':

                print(f"Excel dosyası yükleniyor: {self.veri_yolu}")
                self.df = pd.read_excel(self.veri_yolu)
                print(f"Excel verisi başarıyla yüklendi: {self.df.shape}")
            else:
                raise ValueError(f"Desteklenmeyen dosya formatı: {dosya_uzantisi}")
            
            print(f"Bulunan sütunlar: {list(self.df.columns)}")
            return self.df
            
        except FileNotFoundError:
            print(f" Hata: Dosya bulunamadı: {self.veri_yolu}")
            print("Lütfen veri setinizin doğru konumda olduğundan emin olun:")
            print("   - Excel için: data/Talent_Academy_Case_DT_2025.xlsx")
            return None
        except Exception as e:
            print(f" Veri yükleme hatası: {e}")
            return None
    
    def temel_bilgiler(self):
        print("=" * 60)
        print("Veri seti temel bilgiler")
        print("=" * 60)
        
        print(f"Veri Seti Boyutu: {self.df.shape}")
        print(f"Sütunlar: {list(self.df.columns)}")
        print("\n" + "="*60)
        print("VERİ TİPLERİ")
        print("="*60)
        print(self.df.dtypes)
        
        print("\n" + "="*60)
        print("EKSİK DEĞERLER")
        print("="*60)
        eksik = self.df.isnull().sum()
        eksik_yuzde = (eksik / len(self.df)) * 100
        eksik_df = pd.DataFrame({
            'Eksik Sayısı': eksik,
            'Yüzde': eksik_yuzde
        }).sort_values('Eksik Sayısı', ascending=False)
        print(eksik_df[eksik_df['Eksik Sayısı'] > 0])
        
        print("\n" + "="*60)
        print("İLK 5 SATIR")
        print("="*60)
        print(self.df.head())
    
    def hedef_degisken_analizi(self):
        print("\n" + "="*60)
        print(" Hedef değişken analizi (TedaviSuresi)")
        print("="*60)
        
        hedef = self.df['TedaviSuresi']
        
        print(f"Hedef İstatistikleri:")
        print(f"Sayı: {hedef.count()}")
        print(f"Ortalama: {hedef.mean():.2f}")
        print(f"Medyan: {hedef.median():.2f}")
        print(f"Standart Sapma: {hedef.std():.2f}")
        print(f"Minimum: {hedef.min()}")
        print(f"Maksimum: {hedef.max()}")
        print(f"Eksik: {hedef.isnull().sum()}")
        
        fig, axes = mat.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Hedef Değişken (TedaviSuresi) Analizi', fontsize=16)
        
        axes[0,0].hist(hedef.dropna(), bins=30, alpha=0.7, color='skyblue')
        axes[0,0].set_title('TedaviSuresi Dağılımı')
        axes[0,0].set_xlabel('Tedavi Süresi (Seans)')
        axes[0,0].set_ylabel('Frekans')
        
        axes[0,1].boxplot(hedef.dropna())
        axes[0,1].set_title('TedaviSuresi Kutu Grafiği')
        axes[0,1].set_ylabel('Tedavi Süresi (Seans)')
        
        from scipy import stats
        stats.probplot(hedef.dropna(), dist="norm", plot=axes[1,0])
        axes[1,0].set_title('Q-Q Grafiği (Normal Dağılım)')
        
        en_cok_gorulen = hedef.value_counts().head(15)
        axes[1,1].bar(range(len(en_cok_gorulen)), en_cok_gorulen.values)
        axes[1,1].set_title('En Çok Görülen 15 Değer')
        axes[1,1].set_xlabel('Sıralama')
        axes[1,1].set_ylabel('Sayı')
        
        mat.tight_layout()
        mat.savefig('outputs/figures/hedef_degisken_analizi.png', dpi=300, bbox_inches='tight')
        mat.show()
        
        return hedef.describe()
    
    def kategorik_analiz(self):
        print("\n" + "="*60)
        print(" Kategorik değişkenler analizi")
        print("="*60)
        
        kategorik_sutunlar = ['Cinsiyet', 'KanGrubu', 'Uyruk', 'KronikHastalik', 
                             'Bolum', 'Alerji', 'Tanilar', 'TedaviAdi', 'UygulamaYerleri']
        
        for sutun in kategorik_sutunlar:
            if sutun in self.df.columns:
                print(f"\n {sutun}:")
                print(f"Benzersiz değer sayısı: {self.df[sutun].nunique()}")
                print(f"Eksik değer sayısı: {self.df[sutun].isnull().sum()}")
                
                en_cok_gorulen = self.df[sutun].value_counts().head(10)
                print("En çok görülen 10 değer:")
                for deger, sayi in en_cok_gorulen.items():
                    print(f"  {deger}: {sayi} (%{sayi/len(self.df)*100:.1f})")
    
    def sayisal_analiz(self):
        print("\n" + "="*60)
        print("Sayısal değişkenler analizi")
        print("="*60)
        
        sayisal_sutunlar = ['Yas', 'TedaviSuresi', 'UygulamaSuresi']
        
        if all(sutun in self.df.columns for sutun in sayisal_sutunlar):
            sayisal_df = self.df[sayisal_sutunlar]
            print(sayisal_df.describe())
            
            mat.figure(figsize=(10, 8))
            korelasyon_matrisi = sayisal_df.corr()
            sea.heatmap(korelasyon_matrisi, annot=True, cmap='coolwarm', center=0)
            mat.title('Sayısal Değişkenler Korelasyon Matrisi')
            mat.savefig('outputs/figures/korelasyon_matrisi.png', dpi=300, bbox_inches='tight')
            mat.show()
            
            mat.figure(figsize=(12, 8))
            sea.pairplot(sayisal_df.dropna())
            mat.suptitle('Sayısal Değişkenler Çift Grafiği', y=1.02)
            mat.savefig('outputs/figures/cift_grafik.png', dpi=300, bbox_inches='tight')
            mat.show()
    
    def iliski_analizi(self):
        print("\n" + "="*60)
        print("Hedef değişken ile ilişkiler inceleniyor")
        print("="*60)
        
        mat.figure(figsize=(15, 5))
        
        mat.subplot(1, 3, 1)
        mat.scatter(self.df['Yas'], self.df['TedaviSuresi'], alpha=0.6)
        mat.xlabel('Yaş')
        mat.ylabel('Tedavi Süresi')
        mat.title('Yaş vs Tedavi Süresi')
        
        mat.subplot(1, 3, 2)
        if 'Cinsiyet' in self.df.columns:
            sea.boxplot(data=self.df, x='Cinsiyet', y='TedaviSuresi')
            mat.title('Cinsiyet vs Tedavi Süresi')
        
        mat.subplot(1, 3, 3)
        if 'Bolum' in self.df.columns:
            en_cok_bolum = self.df['Bolum'].value_counts().head(5).index
            alt_kume = self.df[self.df['Bolum'].isin(en_cok_bolum)]
            sea.boxplot(data=alt_kume, x='Bolum', y='TedaviSuresi')
            mat.xticks(rotation=45)
            mat.title('İlk 5 Bölüm vs Tedavi Süresi')
        
        mat.tight_layout()
        mat.savefig('outputs/figures/iliski_analizi.png', dpi=300, bbox_inches='tight')
        mat.show()
    
    def eksik_degerler_isle(self):
        print("\n" + "="*60)
        print("Eksik değerler işleniyor")
        print("="*60)
        
        self.df_islenmis = self.df.copy()
        
        for sutun in self.df_islenmis.columns:
            eksik_sayi = self.df_islenmis[sutun].isnull().sum()
            
            if eksik_sayi > 0:
                print(f"\n{sutun}: {eksik_sayi} eksik değer")
                
                if self.df_islenmis[sutun].dtype == 'object':

                    mod_deger = self.df_islenmis[sutun].mode()
                    if len(mod_deger) > 0:
                        self.df_islenmis[sutun] = self.df_islenmis[sutun].fillna(mod_deger[0])
                        print(f"  Mod ile dolduruldu: {mod_deger[0]}")
                    else:
                        self.df_islenmis[sutun] = self.df_islenmis[sutun].fillna('Bilinmeyen')
                        print(f" 'Bilinmeyen' ile dolduruldu")
                
                else:
                    medyan_deger = self.df_islenmis[sutun].median()
                    self.df_islenmis[sutun] = self.df_islenmis[sutun].fillna(medyan_deger)
                    print(f" Medyan ile dolduruldu: {medyan_deger}")
        
        print(f"\nİşleme sonrası eksik değer sayısı: {self.df_islenmis.isnull().sum().sum()}")
    
    def kategorik_kodla(self):
        print("\n" + "="*60)
        print("Kategorik değişkenler kodlanıyor")
        print("="*60)
        
        kategorik_sutunlar = self.df_islenmis.select_dtypes(include=['object']).columns
        
        if 'HastaNo' in kategorik_sutunlar:
            kategorik_sutunlar = kategorik_sutunlar.drop('HastaNo')
        
        yuksek_kardinalite_sutunlar = []
        for sutun in kategorik_sutunlar:
            benzersiz_sayi = self.df_islenmis[sutun].nunique()
            if benzersiz_sayi > 10:
                yuksek_kardinalite_sutunlar.append(sutun)
        
        label_kodlayicilar = {}
        for sutun in yuksek_kardinalite_sutunlar:
            le = LabelEncoder()
            self.df_islenmis[f'{sutun}_kodlanmis'] = le.fit_transform(self.df_islenmis[sutun].astype(str))
            label_kodlayicilar[sutun] = le
            print(f"Label kodlaması: {sutun} -> {sutun}_kodlanmis")
        
        dusuk_kardinalite_sutunlar = [sutun for sutun in kategorik_sutunlar if sutun not in yuksek_kardinalite_sutunlar]
        
        for sutun in dusuk_kardinalite_sutunlar:
            if self.df_islenmis[sutun].nunique() <= 10:
                dummy_degiskenler = pd.get_dummies(self.df_islenmis[sutun], prefix=sutun)
                self.df_islenmis = pd.concat([self.df_islenmis, dummy_degiskenler], axis=1)
                print(f"One-hot kodlaması: {sutun}")
        
        return label_kodlayicilar
    
    def sayisal_ozellikler_olcekle(self):
        print("\n" + "="*60)
        print("Sayısal özellikler ölçülüyor")
        print("="*60)
        
        sayisal_sutunlar = ['Yas', 'UygulamaSuresi'] 
        
        olcekleyici = StandardScaler()
        for sutun in sayisal_sutunlar:
            if sutun in self.df_islenmis.columns:
                self.df_islenmis[f'{sutun}_olceklenmis'] = olcekleyici.fit_transform(
                    self.df_islenmis[[sutun]]
                )
                print(f"Ölçeklenmiş: {sutun} -> {sutun}_olceklenmis")
        
        return olcekleyici
    
    def model_hazir_veri_uret(self):
        print("\n" + "="*60)
        print("Model hazır veri seti oluşturuluyor")
        print("="*60)
        
        ozellik_sutunlar = []
        
        for sutun in ['Yas_olceklenmis', 'UygulamaSuresi_olceklenmis']:
            if sutun in self.df_islenmis.columns:
                ozellik_sutunlar.append(sutun)
        
        kodlanmis_sutunlar = [sutun for sutun in self.df_islenmis.columns if sutun.endswith('_kodlanmis')]
        ozellik_sutunlar.extend(kodlanmis_sutunlar)
        
        dummy_sutunlar = []
        for sutun in ['Cinsiyet', 'KanGrubu', 'Uyruk']:
            dummy_sutunlar.extend([s for s in self.df_islenmis.columns if s.startswith(f'{sutun}_')])
        ozellik_sutunlar.extend(dummy_sutunlar)
        
        son_ozellikler = ozellik_sutunlar
        hedef_sutun = 'TedaviSuresi'
        
        model_hazir_df = self.df_islenmis[son_ozellikler + [hedef_sutun]].copy()
        
        print(f"Model hazır veri seti boyutu: {model_hazir_df.shape}")
        print(f"Özellik sayısı: {len(son_ozellikler)}")
        print(f"Hedef değişken: {hedef_sutun}")
        
        model_hazir_df.to_csv('outputs/model_hazir_veri_seti.csv', index=False)
        print("Model hazır veri seti kaydedildi")
        
        return model_hazir_df
    
    def tam_analiz_calistir(self):
        print("Tam Veri Analizi Pipeline Başlatılıyor")
        
        os.makedirs('outputs', exist_ok=True)  
        os.makedirs('outputs/figures', exist_ok=True)
        
        if self.veri_yukle() is None:
            return
        
        self.temel_bilgiler()
        self.hedef_degisken_analizi()
        self.kategorik_analiz()
        self.sayisal_analiz()
        self.iliski_analizi()
        
        self.eksik_degerler_isle()
        self.kategorik_kodla()
        self.sayisal_ozellikler_olcekle()
        model_hazir_df = self.model_hazir_veri_uret()

        print("\n" + "="*60)
        print("Analiz tamamlandı")
        print("="*60)
        print(f"Orijinal veri seti: {self.df.shape}")
        print(f"Model hazır veri seti: {model_hazir_df.shape}")
        print("Modellemeye hazır")
        
        return model_hazir_df

def app():
    print("Pusula Veri Bilimi Case Study'si")
    print("="*60)

    excel_yolu = "data/Talent_Academy_Case_DT_2025.xlsx"

    if os.path.exists(excel_yolu):
        veri_yolu = excel_yolu
        print(f"Excel veri seti bulundu: {excel_yolu}")
    else:
        print("Veri seti bulunamadı")
        print("Lütfen veri setinizi şu şekilde yerleştirin:")
        print("   - data/Talent_Academy_Case_DT_2025.xlsx (Excel formatı)")
        return None, None
    
    analizci = VeriAnalizci(veri_yolu)
    model_hazir_veri = analizci.tam_analiz_calistir()
    return analizci, model_hazir_veri

if __name__ == "__main__":
    analizci, model_hazir_veri = app()