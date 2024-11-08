پیش‌بینی مصرف انرژی با استفاده از مدل‌های یادگیری ماشین
این مخزن شامل کدی برای پیش‌بینی مصرف انرژی خانگی بر اساس داده‌های تاریخی با استفاده از مدل‌های مختلف یادگیری ماشین و تکنیک‌های پیش‌بینی سری زمانی است.

پیش‌نیازها
برای اجرای کد، باید وابستگی‌های زیر را نصب کنید:

Python 3.7+
pandas
numpy
statsmodels
scikit-learn
xgboost
matplotlib
ماژول‌های پیش‌بینی
داده‌های پاک‌شده در چهار ماژول مختلف برای پیش‌بینی مصرف انرژی استفاده می‌شوند:

ARIMA (میانگین متحرک یکپارچه خودرگرسیونی): یک مدل کلاسیک سری زمانی که برای پیش‌بینی داده‌های یک‌بعدی با روند و فصلی بودن مناسب است.
رگرسیون خطی: یک مدل ساده رگرسیون که برای پیش‌بینی متغیر هدف (Global_active_power) بر اساس مقادیر گذشته و ویژگی‌ها استفاده می‌شود.
جنگل تصادفی: یک روش یادگیری تجمعی قدرتمند که برای وظایف طبقه‌بندی و رگرسیون استفاده می‌شود و ارتباط بین ویژگی‌ها و متغیر هدف را مدل‌سازی می‌کند.
XGBoost (بوستینگ گرادیانی شدید): یک الگوریتم بوستینگ گرادیانی که به دلیل کارایی و عملکرد بالا در پیش‌بینی وظایف رگرسیون مورد استفاده قرار می‌گیرد و قادر است پیچیدگی‌ها و روابط غیرخطی داده را به‌خوبی شناسایی کند.
نتیجه‌گیری
پس از مقایسه چهار مدل، XGBoost به عنوان بهترین مدل برای پیش‌بینی مصرف انرژی خانگی انتخاب شد و دقیق‌ترین نتایج را ارائه داد. این مدل می‌تواند پیچیدگی‌ها و روابط غیرخطی موجود در داده‌ها را به‌خوبی شبیه‌سازی کند. اما با توجه به داده‌های ورودی و پیچیدگی مساله، ARIMA نیز می‌تواند گزینه قوی‌ای برای پیش‌بینی سری زمانی باشد.
پاک‌سازی داده‌ها
قبل از اعمال مدل‌های یادگیری ماشین یا پیش‌بینی سری زمانی، ابتدا داده‌ها پاک‌سازی و پیش‌پردازش می‌شوند. این مجموعه‌داده شامل اندازه‌گیری‌های مصرف برق خانگی با ستون‌های زیر است:

تاریخ
زمان
Global_active_power
Global_reactive_power
ولتاژ
Global_intensity
Sub_metering_1
Sub_metering_2
Sub_metering_3
مراحل پاک‌سازی داده‌ها:
جایگزینی مقادیر گمشده: مقادیر گمشده با NaN جایگزین و ردیف‌های دارای داده‌های ناقص حذف شدند.
پردازش تاریخ و زمان: ستون‌های تاریخ و زمان ترکیب و به یک ستون datetime تبدیل شدند و به عنوان ایندکس برای تحلیل سری زمانی تنظیم شدند.
تبدیل به اعداد: تمام ستون‌های مرتبط به مقادیر عددی تبدیل شدند تا برای مدل‌سازی قابل استفاده باشند.
نمونه‌برداری ساعتی و روزانه: داده‌ها به صورت ساعتی و روزانه نمونه‌برداری شدند تا دو مجموعه‌داده (df_hourly.csv و df_daily.csv) ایجاد شوند.
