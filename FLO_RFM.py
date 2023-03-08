
###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# GÖREVLER
###############################################################
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# GÖREV 1: Veriyi Anlama (Data Understanding) ve Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.
df_=pd.read_csv(r"C:\Users\sermi\PycharmProjects\pythonProject4\flo_data_20k.csv")
df=df_.copy()

           # 2. Veri setinde
                     # a. İlk 10 gözlem,
df.head(10)
                     # b. Değişken isimleri,
df.columns
                     # c. Betimsel istatistik,
df.describe().T
                     # d. Boş değer,
df.isnull().values.any
                     # e. Değişken tipleri, incelemesi yapınız.
#print([{col: df[col].dtypes} for col in df.columns])
df.dtypes
           # 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
df["order_num_total"]=df["order_num_total_ever_online"]+df["order_num_total_ever_offline"]
df["customer_value_total"]=df["customer_value_total_ever_offline"]+df["customer_value_total_ever_online"]
df.columns
           # 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
#print([{col: df[col].dtypes} for col in df.columns])
#
df_datecols=df.columns[df.columns.str.contains("date", na=False)]
df[df_datecols]=df[df_datecols].apply(pd.to_datetime)
df.dtypes

           # 5. Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların dağılımına bakınız.
df.groupby("order_channel").agg({"master_id": "count", "order_num_total": "sum", "customer_value_total": "sum"})

           # 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
df.groupby("master_id").agg({"customer_value_total": "sum"}).sort_values("customer_value_total", ascending=False).head(10)
df.sort_values("customer_value_total", ascending=False).head(10)

           # 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df.groupby("master_id").agg({"order_num_total": "sum"}).sort_values("order_num_total", ascending=False).head(10)
df.sort_values("order_num_total", ascending=False).head(10)

           # 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.
def preparation_func():

    df_datecols=df.columns[df.columns.str.contains("date", na=False)]
    df[df_datecols]=df[df_datecols].apply(pd.to_datetime)

    df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    
    df.groupby("order_channel").agg({"master_id": "count", "order_num_total": "mean", "customer_value_total": "mean"})
    df.groupby("master_id").agg({"customer_value_total": "sum"}).sort_values("customer_value_total", ascending=False).head(10)
    df.groupby("master_id").agg({"order_num_total": "sum"}).sort_values("order_num_total", ascending=False).head(10)

# GÖREV 2: RFM Metriklerinin Hesaplanması

#print(df.last_order_date.max())
today_date=dt.datetime(2021, 6, 1)
rfm=df.groupby("master_id").agg({
    "last_order_date": lambda date: (today_date - date).dt.days,
    "order_num_total": lambda order_num_total: order_num_total.sum(),
    "customer_value_total": lambda customer_value_total: customer_value_total.sum()
})
rfm.columns=["recency", "frequency", "monetary"]
rfm.describe().T

# GÖREV 3: RF ve RFM Skorlarının Hesaplanması

rfm["recency_score"]=pd.qcut(rfm["recency"], 5, labels=[5,4,3,2,1])
rfm["monetary_score"]=pd.qcut(rfm["monetary"], 5, labels=[1,2,3,4,5])
rfm["frequency_score"]=pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])
rfm["RFM_SCORE"]=(rfm["frequency_score"].astype(str) + rfm["recency_score"].astype(str))
rfm.reset_index(inplace=True)


# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"]=rfm["RFM_SCORE"].replace(seg_map, regex=True)


# GÖREV 5: Aksiyon zamanı!
           # 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg("mean")
           # 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv ye kaydediniz.
                   # a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
                   # tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Sadık müşterilerinden(champions,loyal_customers),
                   # ortalama 250 TL üzeri ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kuralacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına
                   # yeni_marka_hedef_müşteri_id.cvs olarak kaydediniz.

loyal_customers=rfm[rfm["segment"].isin(["loyal_customers", "champions"])]
woman_customers=df[df["interested_in_categories_12"].str.contains("KADIN", na=False)]
yeni_marka_hedef_müşteri_id=pd.merge(woman_customers[["master_id", "interested_in_categories_12"]], loyal_customers[["master_id", "segment"]], on="master_id", how="inner")
yeni_marka_hedef_müşteri_id.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)

                   # b.  Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
                # alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
                # olarak kaydediniz.

target_customers=rfm[rfm["segment"].isin(["new_customers", "about_to_sleep"])]
men_kids_customers=df[(df["interested_in_categories_12"].str.contains("ERKEK", na=False)) | (df["interested_in_categories_12"].str.contains("COCUK", na=False))]

indirim_hedef_müşteri_ids=pd.merge(target_customers[["master_id", "segment"]], men_kids_customers[["master_id", "interested_in_categories_12"]], on="master_id", how="inner")
indirim_hedef_müşteri_ids.to_csv("indirim_hedef_müşteri_ids.csv", index=False)

# GÖREV 6: Tüm süreci fonksiyonlaştırınız.

###############################################################
# GÖREV 1: Veriyi  Hazırlama ve Anlama (Data Understanding)
###############################################################


# 2. Veri setinde
        # a. İlk 10 gözlem,
        # b. Değişken isimleri,
        # c. Boyut,
        # d. Betimsel istatistik,
        # e. Boş değer,
        # f. Değişken tipleri, incelemesi yapınız.



# 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.



# 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.


# df["last_order_date"] = df["last_order_date"].apply(pd.to_datetime)



# 5. Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısı ve toplam harcamaların dağılımına bakınız. 



# 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.




# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.




# 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.


###############################################################
# GÖREV 2: RFM Metriklerinin Hesaplanması
###############################################################

# Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi



# customer_id, recency, frequnecy ve monetary değerlerinin yer aldığı yeni bir rfm dataframe


###############################################################
# GÖREV 3: RF ve RFM Skorlarının Hesaplanması (Calculating RF and RFM Scores)
###############################################################

#  Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çevrilmesi ve
# Bu skorları recency_score, frequency_score ve monetary_score olarak kaydedilmesi




# recency_score ve frequency_score’u tek bir değişken olarak ifade edilmesi ve RF_SCORE olarak kaydedilmesi


###############################################################
# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması
###############################################################

# Oluşturulan RFM skorların daha açıklanabilir olması için segment tanımlama ve  tanımlanan seg_map yardımı ile RF_SCORE'u segmentlere çevirme


###############################################################
# GÖREV 5: Aksiyon zamanı!
###############################################################

# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.



# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulunuz ve müşteri id'lerini csv ye kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Bu müşterilerin sadık  ve
# kadın kategorisinden alışveriş yapan kişiler olması planlandı. Müşterilerin id numaralarını csv dosyasına yeni_marka_hedef_müşteri_id.cvs
# olarak kaydediniz.



# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
# alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.
