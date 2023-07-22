# Görev 1: Verilen değerlerin veri yapılarını inceleyiniz.

x = 8
print(type(x))

y = 3.2
print(type(y))

z = 8j + 18
print(type(z))

a = "Hello World"
print(type(a))

b = True
print(type(b))

c = 23 < 22
print(type(c))

l = [1, 2, 3, 4]
print(type(l))

d = {"Name": "Jake",
     "Age": 27,
     "Address": "Downtown"}
print(type(d))

t = ("Machine Learning", "Data science")
print(type(t))

s = {"Python", "Machine Learning", "Data Science"}
print(type(s))

# Görev 2 : Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz. Kelime kelime ayırınız.

text = "The goal is to turn data into information, and information into insight."

upper = text.upper()
replace = upper.replace(","," ").replace("."," ")
split= replace.split()
print(split)

# Görev 3: Verilen listeye aşağıdaki adımları uygulayınız.

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

# Adım 1: Verilen listenin eleman sayısına bakınız.

print(len(lst))

# Adım 2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.

print(lst[0], lst[10])

# Adım 3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.

print(lst[:4])

# Adım 4: Sekizinci indeksteki elemanı siliniz.

print(lst.remove("N"))
print(lst)

# Adım 5: Yeni bir eleman ekleyiniz.

print(lst.append("S"))
print(lst)

# Adım 6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.

print(lst.insert(8, "N"))
print(lst)

# Görev 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.

dict = {'Christian': ["America", 18],
        'Daisy': ["England", 12],
        'Antonio': ["Spain", 22],
        'Dante': ["Italy", 25]}

# Adım 1: Key değerlerine erişiniz.

print(dict.keys())

# Adım 2: Value'lara erişiniz.

print(dict.values())

# Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.

dict.update({"Daisy": ["England", 13]})
print(dict)

# Adım 4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.

dict.update({"Ahmet":["Turkey", 24]})
print(dict)

# Adım 5: Antonio'yu dictionary'den siliniz.

dict.pop("Antonio")
print(dict)

# Görev 5: Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atayan ve bu listeleri return eden fonksiyon yazınız.


l = [2,13,18,93,22]

def func(lst):
    odd = []
    even = []

    for i in lst:
        if i % 2 == 0:
            even.append(i)

        else:
            odd.append(i)

    return odd, even

odd_list, even_list = func(l)

print(odd_list)
print(even_list)

# Görev 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri bulunmaktadır.
# Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken son üç öğrenci dtıp fakültesi öğrenci sırasına aittir.
# Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.

ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]

for i, student in enumerate(ogrenciler):
    if i < 3:
        print("Mühendislik Fakültesi {}. öğrenci: {}".format(i + 1, student))

    else:
        print("Tıp Fakültesi {}.öğrenci: {}".format(i - 2, student))

# Görev 7: Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi ve kontenjan bilgileri yer almaktadır.
# Zip kullanarak ders bilgilerini bastırınız.

ders_kodu = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]

ders_bilgileri = zip(kredi, ders_kodu, kontenjan)

for kredi, ders_kodu, kontenjan in ders_bilgileri:
    print(f"Kredisi {kredi} olan {ders_kodu} kodlu dersin kontenjanı {kontenjan} kişidir.")

# Görev 8: Aşağıda 2 adet set verilmiştir. Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak elemanlarını
# eğer kapsamıyor ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir

kume1 = set(["data", "python"])

kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

if kume1.issuperset(kume2):
    print(kume1.intersection(kume2))

else:
    print(kume2.difference(kume1))
