# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()

# Görev 2: Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.

passengers = df["sex"].value_counts()
print(passengers)

# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.

num_unique = df[[col for col in df.columns]].nunique()
print(num_unique)

# Görev 4: pclass değişkeninin unique değerlerinin sayısını bulunuz.

pclass_unique = df["pclass"].value_counts()
print(pclass_unique)

# Görev 5: pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.

pclass_parch = df[["pclass", "parch"]].nunique()
print(pclass_parch)

# Görev 6: embarked değişkeninin tipini kontrol ediniz.
# Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.

print(df["embarked"].dtypes)
df["embarked"] = df["embarked"].astype("category")
print(df["embarked"].dtype)

# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.

embarked_c = df[df["embarked"] == "C"]
embarked_c.head()

# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.

embarked_not_s = df[df["embarked"] != "S"]
pd.set_option("display.max_columns", None)
embarked_not_s.head()

# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.

age_and_sex = df[(df["age"] < 30) & (df["sex"] == "female")]
print(age_and_sex)

# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.

fare_or_age = df[(df["fare"] > 500)| (df["age"] > 70)]
fare_or_age.head()

# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.

df.isnull().sum()

# Görev 12: who değişkenini dataframe’den çıkarınız.

drop_who = df.drop("who",axis=1)
drop_who.head()

# Görev 13: deck değişkenindeki boş değerleri deck değişkeninin en çok tekrar eden değeri (mode) ile doldurunuz.

df["deck"].fillna(df["deck"].mode()[0], inplace=True)

# Görev 14: age değikenindeki boş değerleri age değişkeninin medyanı ile doldurunuz.

df["age"].fillna(df["age"].median())

# Görev 15: survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.

df.groupby(["pclass", "sex"]).agg({"survived": ["sum", "count", "mean"], })

# Görev 16: 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz.
# (apply ve lambda yapılarını kullanınız)

def check_age(x):
    if x["age"] < 30:
        return 1
    else:
        return 0
df["age_flag"] = df.apply(lambda x: check_age(x), axis=1)
print(df)

# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.

df_tips = sns.load_dataset("tips")
print(df_tips)

# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerlerinin
# toplamını, min, max ve ortalamasını bulunuz.

print(df_tips.groupby("time").agg({"total_bill":["sum", "min", "max", "mean"]}))

# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.

print(df_tips.groupby(["day", "time"]).agg({"total_bill":["sum", "min", "max", "mean"]}))

# Görev 20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre
# toplamını, min, max ve ortalamasını bulunuz.

print(df_tips.loc[(df_tips.time == "Lunch") & (df_tips.sex == "Female")].groupby("day").agg(
    {"total_bill": ["sum", "min", "max", "mean"], "tip": ["sum", "min", "max", "mean"]}))

# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)

print(df_tips.loc[(df_tips["size"] < 3) & (df_tips["total_bill"] > 10)].agg({"total_bill": "mean"}))

# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturunuz.
# Her bir müşterinin ödediği totalbill ve tip in toplamını versin.

df_tips["total_bill_tip_sum"] = df_tips["total_bill"] + df_tips["tip"]
print(df_tips)

# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız
# ve ilk 30 kişiyi yeni bir dataframe'e atayınız.

new_df = df_tips.sort_values("total_bill_tip_sum", ascending=False).head(30)
print(new_df)
