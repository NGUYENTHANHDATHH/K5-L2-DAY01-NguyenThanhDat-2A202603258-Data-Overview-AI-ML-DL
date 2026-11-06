# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= "cab" ; `rank`=1 ; `score`= 0.510915; `taxonomy_name`="ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   Record này nói về một vật thể riêng trong bức ảnh. Vì số lượng ảnh chỉ có 3 nhưng số lượng phân loại mà mô hình dự đoán là 5, cho thấy trong ảnh có thể có nhiều hơn một vật thể.
3. Danh sách tên lớp (class list) do đâu mà có: danh sách tên lớp do taxonomy/checkpoint quyết định.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   Vì do mỗi vật thể là riêng biệt nên cần id để phân biệt, cần taxonomy_name vì mỗi lớp cũng có thể phân loại ra cùng tên vật thể nên cần tên lớp để phân biệt loại taxonomy.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp? Guideline cần quy định quy tắc phân xử khi nhiều lớp trong taxonomy cùng phù hợp
6. Vì sao không được lấy `score` của model làm ground truth? vì score chỉ thể hiện mức độ tự tin của model đối với dự đoán, không chứng minh dự đoán đó đúng với thực tế.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="cup"; `score`=0.381542 ; `bbox_xyxy`=[119.95,272.2, 141.78,304.16] ; `bbox_width`=21.84 ; `bbox_height`= 31.96
2. Giải thích box bằng lời thường: góc trên-trái (119.95, 272.2), góc dưới-phải (141.78, 304.16), rộng 21.84 pixel, cao 31.96 pixel.
3. Chọn hai ngưỡng: Ngưỡng 0.2 có 17 prediction, Ngưỡng 0.60 có 6 prediction
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), tức là giữ lại nhiều hộp có độ tin cậy thấp hơn nên reviewer sẽ phải xác minh nhiều hộp có khả năng sai hơn.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này): với mỗi taxonomy cần gán nhãn, annotator phải vẽ hình chữ nhật bao quanh diện tích đối tượng nhìn thấy được, không được cắt mất một phần của đối tượng, hạn chế tối đa vùng không liên quan.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao? Có thể đưa vào guideline nếu vật thể vẫn có thể nhận diện rõ và áp dụng cho nhiều ảnh. Giúp cho các annotator xử lý nhất quán, giảm nguy cơ thiếu nhãn ảnh hưởng đến ground truth của yolo11. Nếu vật thể bị che khuất quá nhiều, guideline hiện tại chưa đủ quy định cách xử lý thì cần escalation lên lab coach.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`="kitchen-010"; `class_name`="bowl" ; `score`=0.372406 ; số điểm polygon=`polygon_point_count`12 ; 3–5 điểm đầu của `polygon_xy`=[[155.0,170.0],[155.0,177.0],[156.0,178.0],[156.0,179.0]]
2. So với box, polygon cho thêm thông tin về lớp bao quanh vật thể chi tiết hơn về hình dạng vật thể
3. `instance_id` dùng để chỉ ID trong output lab. Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào? instance_id dùng để định danh từng đối tượng riêng lẻ trong output lab. Nhiều đối tượng có thể cùng class_id nhưng phải có instance_id khác nhau trong phạm vi quy định.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo: Annotator phải vẽ đường biên mask bám sát đường viền thực tế của phần đối tượng nhìn thấy được, bao phủ toàn bộ pixel thuộc đối tượng và hạn chế tối đa pixel nền. Không vẽ lấn sang vật thể khác, không bỏ sót các phần nhìn thấy và không tự suy đoán phần bị che khuất hoặc nằm ngoài ảnh.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao? Tôi sẽ kiểm tra guideline trước. Nếu đã có quy tắc rõ ràng thì áp dụng; nếu ranh giới mơ hồ và chưa có quy tắc xử lý thì escalation lên Lab Coach. Vì Ground Truth cần nhất quán, không nên tự suy đoán đường biên mask khi thiếu căn cứ.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                                                                     | Annotator làm gì?                                                     | Reviewer xem gì?                                                           |
| --------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Phân loại ảnh         | Một ảnh                                       | Ảnh traffic có chứa nhiều vật thể như con người, xe cộ khác nhau, khiến việc xác định một class chính trở nên khó khăn | Xác định ảnh thuộc class nào trong taxonomy. Gán nhãn cho toàn bộ ảnh | Class có đúng không? Có vi phạm quy tắc chọn lớp chính không?              |
| Phát hiện vật thể     | Một đối tượng                                 | Trên ảnh kitchen, chiếc oven thứ 2 bị cắt mép, khiến việc xác định bounding box khó khăn                               | Vẽ bounding box và gán class                                          | Có thiếu box , box thừa , sai class, box lệch không?                       |
| Instance segmentation | Một đối tượng                                 | Trên ảnh kitchen, hai oven chạm nhau, khiến việc xác định ranh giới mask khó khăn                                      | Vẽ mask cho từng vật thể                                              | Mask có bám sát biên không? Có lấn nền, thiếu vùng, gộp hai vật thể không? |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? chỉ sử dụng dữ liệu trong phạm vi công việc và trên công cụ được cho phép; không tự ý sao chép, tải xuống hoặc chia sẻ ảnh, JSON và Ground Truth ra bên ngoài.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai? Tôi sẽ dừng và báo lên lab coach hoặc người phụ trách dự án qua kênh được quy định

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
      Beta
      AI can make mistakes, so double-check it.
      0 / 0
      used queries
      1
