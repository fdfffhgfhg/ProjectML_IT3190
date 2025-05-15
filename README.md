# ProjectML_IT3190
Dự án bài tập lớn môn Nhập môn học máy và Khai phá dữ liệu (IT3190) . Đề tài : Nhận diện chữ viết tay .

Xin chào tất cả các bạn , đây là mã nguồn của dự án nhận diện chữ viết tay (Text Recognition) của nhóm 17 , lớp 157320 , môn nhập môn học máy và khai phá dữ liệu(IT3190) . Mã nguồn trên được nhóm chúng tôi lập trình và triển khai trên nền tảng VSCode, hệ điều hành Windows , ngôn ngữ lập trình Python và dùng các thư viện học máy phổ biến như keras , tensorflow , ... . Phiên bản python và các thư viện cần thiết đều đã được đề cập ở trên file requirement.txt:

- PyYAML>=6.0
- tqdm
- qqdm==0.0.7
- pandas
- numpy
- opencv-python
- Pillow>=9.4.0
- onnxruntime>=1.15.0  # onnxruntime-gpu for GPU support
- matplotlib
- Python = 3.8.2
- mltu = 1.1.8
- tensorflow = 2.10.0


Các thư viện cần thiết đều có thể được dễ dàng cài đặt qua lệnh pip trên terminal . Ví dụ , để tải thư viện tensorflow 2.10.0 , ta dùng lệnh :

     pip install tensorflow==2.10.0 .

Để kiểm tra các thư viện đã được cài vào môi trường , bạn có thể dùng dòng lệnh pip list trên terminal.


Trong dự án này , chúng tôi xây dựng và đánh giá mô hình trên tập dữ liệu IAM Word Dataset . Trước hết , các bạn chạy file downloadDataset.py ở thư mục Source để tải tập dữ liệu . Tập dữ liệu và các file liên quan sẽ được giải nén và lưu trữ trong thư mục Datasets . Dưới đây là mã nguồn để tải tập dữ liệu: 

```python
     import os
     import tarfile
     from tqdm import tqdm
     from urllib.request import urlopen
     from io import BytesIO
     from zipfile import ZipFile
     def download_and_unzip(url, extract_to="Datasets", chunk_size=1024*1024):
           http_response = urlopen(url)

           data = b""
           iterations = http_response.length // chunk_size + 1
           for _ in tqdm(range(iterations)):
           data += http_response.read(chunk_size)

          zipfile = ZipFile(BytesIO(data))
         zipfile.extractall(path=extract_to)

     dataset_path = os.path.join("Datasets", "IAM_Words")
     if not os.path.exists(dataset_path):
     download_and_unzip("https://git.io/J0fjL", extract_to="Datasets")

    file = tarfile.open(os.path.join(dataset_path, "words.tgz"))
    file.extractall(os.path.join(dataset_path, "words"))
```

Tổng quan về tập dữ liệu IAM Word Dataset : đây là một tập dữ liệu gồm các ảnh chữ viết tay đơn lẻ và mỗi ảnh sẽ có một nhãn tương ứng với nó . Có tổng cộng 115338 ảnh từ đơn , với các nhãn được gán trong file words.txt . Bộ dữ liệu này cung cấp các ví dụ rất đa dạng về các kí tự : 26 chữ cái tiếng anh thường , 26 chữ cái tiếng anh viết hoa , 10 chữ số từ 0 - 9 và 16 kí tự đặc biệt (.,()'":) , vậy nên có thể bước đầu xác định rằng bài toán thực hiện trên tập dữ liệu này sẽ có 78 nhãn. Trong file text cũng gồm khoảng 116000 dòng , cấu trúc của mỗi dòng như sau : 

```text
    a01-000u-00-05 ok 154 1438 746 382 73 NP Gaitskell

    a01-000u-00-05 : Chỉ đường dẫn tới ảnh trong dataset
    ok : Nhãn thông báo rằng ảnh này là tốt về mặt chất lượng (err nếu ngược lại)
    154 : Ngưỡng chuẩn hóa nhị phân của ảnh . Trong các bài toán xử lý ảnh xám , người ta sẽ chuẩn hóa các pixel về toàn trắng (255) hoặc toàn    đen (0) . Nhóm chúng tôi không sử dụng thông tin này trong tiền xử lý dữ liệu . 
    1438 746 382 73 : Tọa độ bounding box của chữ trong ảnh.
    NP : Nhãn ngữ pháp cho từ trong ảnh , thông tin này cũng không được sử dụng trong huấn luyện mô hình.
    Gaitskell : nhãn tương ứng với chữ trong ảnh.
```



Với các dòng có nhãn 'ok' thì hoàn toàn có thể được đưa vào trong dataset để huấn luyện . Trong tập dữ liệu này , có tới gần 19000 ảnh bị gắn nhãn 'err' , và việc đưa những ảnh lỗi này vào trong quá trình huấn luyện có rất nhiều rủi ro : Giảm độ chính xác của mô hình , Overfitting cho nhãn sai , làm giảm độ tin cậy của mô hình ... Do đó , nhóm chúng tôi sẽ sử dụng tổng cộng là 96456 ảnh được gán nhãn 'ok' để đưa vào dataset .

Dataset của chúng tôi là một danh sách các cặp (x,y) trong đó : x là đường dẫn tới ảnh trong tập dữ liệu IAM , y là nhãn được lấy ra từ file word.text . Với một ảnh , giả dụ như ví dụ trên , ảnh a01-000u-00-05 nằm tại đường dẫn a01/a01-000u/a01-000u-00-05.png , tức là ta hoàn toàn có thể truy cập được tới ảnh thông qua đường dẫn của ảnh trong file text , bằng một số thao tác trên chuỗi . Dưới đây là mã nguồn tạo dataset của chúng tôi , mã nguồn này tạo ra dataset bằng cách đọc từng dòng trên file word.text : 
```python
      dataset, vocab, max_len = [], set(), 0
      words = open(os.path.join(dataset_path, "words.txt"), "r").readlines()
      for line in tqdm(words):
          if line.startswith("#"):
               continue

       line_split = line.split(" ")
       if line_split[1] == "err":
           continue

       folder1 = line_split[0][:3]
       folder2 = "-".join(line_split[0].split("-")[:2])
       file_name = line_split[0] + ".png"
       label = line_split[-1].rstrip("\n")

       rel_path = os.path.join(dataset_path, "words", folder1, folder2, file_name)
       if not os.path.exists(rel_path):
           print(f"File not found: {rel_path}")
          continue

       dataset.append([rel_path, label])
       vocab.update(list(label))
       max_len = max(max_len, len(label))
```

Như các bạn thấy , trong mã nguồn trên , ngoài việc cập nhật dataset , chúng tôi còn cập nhật thêm vocab gồm các kí tự xuất hiện trên các nhãn và max_len là độ dài nhãn lớn nhất . Chúng tôi sẽ trình bày chi tiết về việc các biến này sẽ được sử dụng như thế nào trong bước xây dựng mô hình.

Chúng tôi thiết kế cấu hình của model trong file config.py , tại đó chúng tôi tạo một lớp ModelConfigs lưu các thông số của mô hình :
```text
        self.model_path = os.path.join("Models/", datetime.strftime(datetime.now(), "%Y%m%d%H%M")) : Trong quá trình huấn luyện , các thông tin    của model(log , cập nhật tham số , ...) sẽ được lưu trong thư mục Models với tệp tên là nhãn thời gian lúc bắt đầu train.
        self.height = 32 , self.width = 128 : Kích thước của ảnh được resize khi train mô hình . Thường thì kích thước của ảnh sẽ được resize về (32,128) trước khi được đưa vào huấn luyện trong mạng CNN.
        batch_size : Số lượng ảnh được xử lý một lúc khi huấn luyện , nếu như giá trị này nhỏ thì việc training có thể bị chậm , nhưng nếu quá lớn thì sẽ có thể bị tràn RAM . Ở đây , chúng tôi để batch_size = 512.
        learning_rate : Tốc độ học của mô hình.
        epoch : Số lần mô hình sẽ lặp qua toàn bộ tập dữ liệu huấn luyện trong toàn bộ quá trình huấn luyện.
        worker : Số luồng (CPU workers) dùng để load dữ liệu song song trong quá trình huấn luyện.
        vocab : Là danh sách các kí tự mà mô hình có thể dự đoán được . Chúng tôi lưu thông tin của biến vocab (ở đoạn code ở trên) vào thuộc tính này , do mô hình chỉ dự đoán các ảnh trong tập IAM Word.
        max_text_length : Độ dài tối đa của nhãn mà mô hình sẽ xử lý . Chúng tôi gán giá trị nhãn dài nhất trong tập dữ liệu IAM (là 21) vào thuộc tính này. Tất nhiên , trong thực tế sẽ có những từ có độ dài lớn hơn 21 rất nhiều -> việc dự đoán sai , nhưng việc giới hạn đầu ra của nhãn dự đoán cũng có những ưu điểm nhất định : giúp tăng tốc huấn luyện , tăng tính ổn định của mô hình , giảm chi phí tính toán ...
````
Với dataset thu được ở trên , chúng tôi chuyển nó vào đối tượng DataProvider : đây là 'dataset thực' mà chúng tôi sẽ đưa vào mô hình huấn luyện . Chúng tôi chia tập dữ liệu thành 3 phần : train:validate:test với tỷ lệ 90:5:5 (hoặc có thể là 80:10:10) .
   Với ảnh (images), chúng tôi resize chúng về cùng một kích thước cố định (32*128) và đối với riêng tập train , chúng tôi thực hiện các phép tăng cường dữ liệu như : Chỉnh độ sáng , Chỉnh độ dày của chữ , Chỉnh độ sắc của ảnh , Xoay ảnh . Các phép tăng cường dữ liệu nhằm giúp cho dữ liệu được đa dạng hóa và gần với dữ liệu thực tế hơn.
   Với các nhãn (label) , chúng tôi thực hiện LabelIndexing đối với chuỗi để chuyển thành chuỗi chỉ số , dùng chuỗi vocab trong lớp config làm đối chiếu . Sau đó , với mỗi vecto chỉ số , chúng tôi thực hiện LabelPadding , gán thêm nhãn chỉ số để cho mỗi vecto đều có một kích thước cố định - ở đây chính là độ dài của max_text_length. Các nhãn mới đều mang một giá trị blank (có thể là -1 , 100) , nhưng ở đây chúng tôi sẽ cài mặc định là độ dài của chuỗi vocab , và điều này không gây ảnh hưởng gì cả.

Về thiết kế Model.
Bài toán từ đầu được nhóm chúng tôi xác định chính là nhận diện chữ viết tay , đây là bài toán thuộc lớp bài toán nhận dạng chuỗi , thuộc kiểu bài toán học có giám sát . Giải pháp mà chúng tôi áp dụng cho bài toán này chính là thiết kế một mạng CNN + RNN(Bi-LSTM) + CTC . Đây là phương pháp hiện đại dùng để giải quyết bài toán nhận dạng chuỗi , với rất nhiều ưu điểm :

- Không cần phải phân đoạn ảnh để lấy từng kí tự.
- Xử lý tốt với các trường hợp chữ dính liền
- Xử lý được từ với độ dài không cố định
- Việc cài đặt tuy phức tạp , nhưng đảm bảo độ chính xác và tốc độ cao.

(Ô Hoàng Edit phần thiết kế mô hình vào đây)

Đánh giá mô hình trong khi huấn luyện 

Để đánh giá hiệu suất và độ chính xác của mô hình , chúng tôi sử dụng các độ đo (metric) như : CER(Character Error Rate) , WER(Word Error Rate) , loss . Trong khi lặp qua tập luyện ở mỗi epoch khi huấn luyện mô hình , giá trị này sẽ liên tục được cập nhật.

Giải thích về các độ đo : 

     CER(Character Error Rate) : Là tỷ lệ kí tự sai . Công thức tính CER cho một nhãn dự đoán so với nhãn thật của ảnh là : S+D+I/N . Trong đó:
     
     - S : Số kí tự bị thay thế so với nhãn
     - I : Số kí tự thêm vào (kí tự thừa)
     - D : Số kí tự bị mất đi
     - N : Tổng số kí tự ở nhãn gốc
     
     WER(Word Error Rate) về cách tính giống như CER , nhưng phạm vi xét được mở rộng lên thành từ.
     
     Loss : Giá trị mà hàm CTC Loss trả về
     
     Mô hình học sai hoàn toàn khi giá trị CER , WER >= 1 và đúng hoàn toàn khi CER,WER = 0

Nhìn chung , cả 3 giá trị này càng thấp thì mô hình học càng chính xác . Sau mỗi epoch khi mà mô hình lặp qua tập dữ liệu một lần , nó sẽ cập nhật các giá trị loss,CER,WER,val_loss(loss trên tập validation),val_CER(CER trên tập validation),val_WER(WER trên tập validation) trong file log.log

(Đánh giá mô hình sau khi đã huấn luyện)

Tuy rằng đã rất cố gắng trong quá trình tìm hiểu và xây dựng mô hình , và đã thu được kết quả khá khả quan , thế nhưng mô hình của chúng tôi sẽ có những thiếu xót . Chúng tôi xin phép đề cập các giải pháp để nâng cấp mô hình , làm cho mô hình dự đoán chính xác hơn :
- Một mô hình có thể học tốt khi tập dữ liệu mà nó sử dụng để train đủ lớn và đủ độ bao quát. Do vậy , một cách để nâng cao hiệu suất của mô hình chính là huấn luyện nó trên một tập dữ liệu lớn hơn và bao quát hơn tập IAM Word.
- Tăng kích thước mô hình bằng cách tăng số lớp ẩn hoặc tăng số lượng tham số trong mô hình . Điều này giúp cho mô hình có thể "học" được các đặc trưng kĩ càng hơn , giúp gia tăng độ chính xác cho mô hình.
- Thay đổi learning rate : Trong quá trình huấn luyện , chúng tôi liên tục thử nghiệm với các giá trị Learning rate khác nhau : 0.0005,0.001,0.002 . Nhìn chung , tốc độ học lớn hơn giúp mô hình hội tụ nhanh hơn , nhưng chúng tôi thích việc đặt learning rate thấp vì mô hình học tuy chậm và cần nhiều epoch hơn, nhưng mô hình học rất ổn định.
- Áp dụng thêm các kĩ thuật tăng cường dữ liệu : Trong bài toán này , chúng tôi đã sử dụng các kĩ thuật tăng cường dữ liệu như Tăng độ dày của chữ , Xoay ảnh , Chỉnh độ sáng ... Tuy nhiên , các bạn vẫn có thể sử dụng các kĩ thuật khác như : Thêm nhiễu , Tăng độ tương phản ... Tăng cường dữ liệu là một công đoạn quan trọng trong xử lý dữ liệu , nó giúp giảm overfitting cho mô hình , tăng độ bao quát của mô hình mà không cần phải thu thập quá nhiều dữ liệu.

Cảm ơn các bạn đã dành thời gian đọc . Bạn có thể tải , trải nghiệm và phát triển mô hình theo hướng của riêng mình hoàn toàn miễn phí . Hy vọng rằng tài liệu này có thể giúp bạn trong con đường học tập của mình.
