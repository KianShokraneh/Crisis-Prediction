# توضیح کوتاه آزمایش‌ها

در این پروژه، به‌دلیل نیاز Twitter API به اشتراک/هزینه، مرحله‌ی دریافت داده از توییتر با **داده‌ی آفلاین (Sentiment140)** جایگزین شد. همچنین به‌دلیل هزینه‌ی فراخوانی ChatGPT/OpenAI API، بخش برچسب‌گذاری/طبقه‌بندی مبتنی بر API با یک **مدل آفلاین (zero-shot محلی)** جایگزین شده است.

این پروژه دو بخش آزمایشی اصلی دارد: (۱) ساخت داده‌ی «bursty» و اجرای پایپ‌لاین تشخیص سیگنال/برست، (۲) ارزیابی چند مدل احساس‌سنجی روی Sentiment140.

## ۱) ساخت داده و اجرای پایپ‌لاین (OSOS)

- **داده‌ی ورودی:** `data/Scrap_Results.csv` (از Sentiment140) و نسخه‌ی bursty آن `data/Scrap_Results_bursty.csv`.
- **تنظیمات ساخت برست (اسکریپت `Modules/OSOS/create_artificial_bursts.py`):**
  - `start-date=2009-04-01` ، `days=365`
  - `burst-days=3` با `burst-lengths=5,10,20`
  - `burst-multiplier=4`
  - `seed=42`
- **تنظیمات طبقه‌بندی دوکلاسه (داخل `Modules/OSOS/Full_pipeline.py`):**
  - مدل zero-shot: `valhalla/distilbart-mnli-12-1`
  - برچسب‌ها: `informative` / `not_informative`
  - داده برای سرعت به ۴ بخش تقسیم و پردازش می‌شود.
- **تنظیمات تشخیص برست (داخل `Modules/OSOS/Full_pipeline.py`):**
  - آستانه: `threshold=4` (افزایش تعداد توییت روزانه)
  - حداقل طول: `min_length=2` روز
- **خروجی‌ها:** نمودارهای برست در `data/burst_plot_*.png` و بازه‌های برست در `data/bursts_output.csv`.
- **تعداد اجرا:** هر اجرا یک‌بار با همین تنظیمات انجام می‌شود (برای تکرارپذیری از `seed` استفاده شده است).

## ۲) ارزیابی مدل‌های احساس‌سنجی 

- **اسکریپت:** `Modules/OSOS/task1_eval_models.py`
- **تنظیمات اصلی (پیش‌فرض اسکریپت):**
  - `seed=42`
  - `average=macro`
  - `batch-size=32` ، `max-length=128`
  - مدل‌ها: DistilBERT (SST-2)، BERT (SST-2)، RoBERTa (twitter-roberta)
- **معیارها:** Accuracy، Precision، Recall، F1 (مطابق فایل جدول).
- **خروجی‌ها:** نتایج خام در `data/evaluation_results.csv` و جدول خلاصه در `data/evaluation_table.md`
