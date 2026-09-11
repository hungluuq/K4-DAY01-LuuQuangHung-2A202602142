# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/9/2026

**Runtime Colab:** GPU

**Python: 3.13.15/ PyTorch: 2.11.0+cu128/ Ultralytics: 8.4.145**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
	(`468`, `cab`, `1`, `0.510915`, `ImageNet-1K`)
- Record này mô tả toàn ảnh như thế nào?
	Record cho mẫu traffic mô tả toàn bộ ảnh bằng cách đưa ra các nhãn phân loại tiềm năng nhất cho bức hình đó.
	Kết quả dự đoán hàng đầu (Rank 1) là 'cab' với độ tin cậy khoảng 0.51. Ngoài ra, mô hình cũng đưa ra các lựa chọn khác trong Top-5 như 'minibus', 'police_van', 'recreational_vehicle' và 'streetcar'. Điều này cho thấy record này không chỉ định danh một vật thể duy nhất mà đang cố gắng gán một nhãn tổng quát phù hợp nhất với nội dung chủ đạo của toàn bộ khung cảnh giao thông trong ảnh.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
	"taxonomy_name": "ImageNet-1K" được sử dụng để huấn luyện checkpoint yolo11n-cls.pt này.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
	Việc lưu trữ đồng thời taxonomy_name, class_id và class_name trong dữ liệu dự đoán AI là yếu tố then chốt để đảm bảo tính toàn vẹn dữ liệu nhờ ba ưu điểm cốt lõi:
	Taxonomy Name: Xác định rõ "ngôn ngữ" hoặc nguồn bộ dữ liệu huấn luyện (ImageNet-1K), loại bỏ nguy cơ nhầm lẫn khi các dataset khác nhau có chung một mã định danh ID.
	Class ID: Tối ưu hóa hiệu suất tính toán và lập trình, giúp máy tính lọc, sắp xếp và xử lý số liệu nhanh chóng, chính xác mà không gặp lỗi định dạng văn bản.
	Class Name: Mang lại khả năng đọc hiểu trực quan cho con người, hỗ trợ đắc lực cho kỹ sư và kiểm thử viên trong việc gán nhãn, kiểm tra chất lượng và xây dựng guideline.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
	Chủ thể chính (Dominant Object): Ưu tiên vật thể chiếm diện tích lớn nhất hoặc nằm ở vị trí trung tâm.
	Ngữ cảnh/Hành vi (Context): Ưu tiên khái niệm bao quát hành động tổng thể
	Thứ tự ưu tiên (Hierarchy): Áp dụng bảng phân cấp định sẵn khi các vật thể có độ nổi bật ngang nhau 
	Thiết lập này triệt tiêu tính cảm tính của người gán nhãn, bảo đảm dữ liệu đầu ra đạt độ nhất quán (consistency) tuyệt đối.
- Vì sao model score không phải ground truth?
	Model score không phải là ground truth vì: Score là "ý kiến" của máy, Ground Truth là "sự thật" được kiểm chứng.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
	"class_name": "person",
	"score": 0.912625,
	"bbox_xyxy": [
		385.33,
		69.24,
		498.92,
		348.92
	],
	"bbox_width": 113.58,
	"bbox_height": 279.68
- Diễn giải vị trí box bằng lời:
	Vị trí box bao quanh chủ thể với hình chữ nhật
- So sánh số prediction ở hai threshold:
	Threshold = 0.20 (17 vật thể): Bắt được nhiều đối tượng nhỏ, bị che khuất hoặc có độ tin cậy thấp (thìa, cốc, chai, chậu cây).
	Threshold = 0.60 (6 vật thể): Chỉ giữ lại các đối tượng có độ tin cậy cao gồm 2 person, 2 bowl và 2 oven.
	Tăng threshold làm giảm mạnh số lượng dự đoán (từ 17 xuống 6), ưu tiên độ chắc chắn thay vì độ phủ.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
	Độ bao phủ :
		Ngưỡng 0.20 (Cao - 17 vật thể): Bắt trọn các chi tiết nhỏ/ở xa (thìa, cốc, chậu cây), hạn chế bỏ sót nhưng tăng tỷ lệ dự đoán sai (nhiễu).
		Ngưỡng 0.60 (Thấp - 6 vật thể): Bỏ qua hầu hết vật thể nhỏ, chỉ giữ lại đối tượng lớn và rõ ràng (người, lò nướng).
	Khối lượng việc của Reviewer :
		Ngưỡng 0.20: Khối lượng tăng gần gấp 3; reviewer tốn thời gian lọc và xóa các nhãn sai .
		Ngưỡng 0.60: Số hộp cần duyệt ít, nhưng reviewer phải tốn công tự vẽ bổ sung các vật thể bị mô hình bỏ sót .
- Đề xuất một quy tắc box chặt:
	Ôm sát đối tượng: Đường viền áp sát tối đa các cạnh ngoài của đối tượng mục tiêu, hạn chế khoảng trống thừa và không chứa nền hay vật thể khác.
	Lấy trọn phần hiển thị: Bao gồm tất cả các phần nhìn thấy được của đối tượng, kể cả khi đối tượng bị che khuất một phần.
	Giới hạn theo khung hình: Dừng viền tại mép ảnh nếu đối tượng bị tràn/cắt ra ngoài rìa ảnh.
	Hạn chế chồng chéo: Không để các hộp giới hạn đè lên nhau, trừ khi các đối tượng ngoài thực tế thực sự lấn hoặc che khuất lẫn nhau.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
	3 vấn đề cốt lõi cần Guideline hoặc Escalation quyết định:
		Bị che khuất (Occlusion): Cần quy định rõ tỷ lệ hiển thị tối thiểu để quyết định dán nhãn hay bỏ qua, tránh gây nhiễu dữ liệu huấn luyện khi vật thể lộ diện quá ít.
		Bị cắt mép ảnh : Cần thống nhất quy tắc viền hộp phải dừng chính xác tại biên ảnh , và quy định mức độ cắt tối đa được phép trước khi buộc phải hủy nhãn do mất khả năng nhận diện.
		Chồng lấn nhiều vật thể : Cần văn bản hướng dẫn rõ việc đánh dấu từng đối tượng riêng lẻ hay gán nhãn chung cả nhóm/cụm .

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
	"instance_id": "traffic-001",
	"class_name": "bus",
	"score": 0.925745,
	"polygon_point_count": 120,
	"polygon_xy": [[148.0,189.0], [147.0,190.0], [145.0,190.0], ...]
- Polygon bổ sung chi tiết gì so với box?
	Polygon cung cấp ranh giới chính xác hơn hẳn so với Bounding Box
- `instance_id` dùng để làm gì và không phải loại ID nào?
	Dùng để phân biệt các cá thể khác nhau trong cùng một loại . Nó không phải là class_id và không nhất thiết phải duy nhất trên toàn bộ tập dữ liệu .
- Đề xuất một quy tắc biên mask:
	Mask phải bám sát ranh giới hình ảnh rõ ràng. Với phần bị che khuất, cần mở rộng mask theo hình dạng dự kiến của vật thể (suy luận logic) thay vì chỉ bao quanh phần nhìn thấy được.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
	Vùng bị mờ hoặc vật thể có độ phân giải quá thấp.Các vật thể chồng lấn, tiếp xúc phức tạp.Quy định về mức độ che khuất tối thiểu để bắt đầu gán nhãn.


## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Single label/ID toàn ảnh (class_id, class_name, taxonomy_name) | Ảnh chứa nhiều chủ thể cùng lúc; phân vân giữa nhãn tổng thể (bối cảnh) và từng vật thể con | Chọn đúng một nhãn duy nhất theo quy tắc ưu tiên (chủ thể chính/bối cảnh/phân cấp) | Kiểm tra tính nhất quán, độ chính xác nhãn so với guideline phân cấp |
| Phát hiện vật thể | Bounding box 2D ([xmin, ymin, xmax, ymax] hoặc [x, y, w, h]) kèm class label | Bỏ sót vật thể nhỏ/xa; nhầm lẫn giữa các lớp tương đồng (bus vs car/truck); hộp bao không khít hoặc chồng lấn | Vẽ/điều chỉnh hộp bao ôm sát biên vật thể; gán đúng nhãn cho từng đối tượng tách biệt | Kiểm tra độ phủ (tránh sót vật thể), độ khít của box, lọc box dư thừa/trùng lặp |
| Instance segmentation | Polygon (tập hợp tọa độ đỉnh [[x, y], ...]) kèm class label cho từng cá thể | Ranh giới phức tạp khi vật thể bị che khuất; biên đa giác quá lỏng hoặc cắt lẹm vào chi tiết | Chấm điểm viền pixel khít ranh giới thực tế; tách rời các thực thể chạm nhau | Kiểm tra độ khép kín của polygon, độ chính xác đường biên (IoU), phân tách cá thể đúng chuẩn |

## 5. An toàn dữ liệuChọn đúng một nhãn duy nhất theo quy tắc ưu tiên (chủ thể chính/bối cảnh/phân cấp)

- Một quy tắc bảo vệ dữ liệu:
	Giới hạn mục đích (Purpose Limitation): Dữ liệu chỉ được thu thập cho mục đích cụ thể, rõ ràng, hợp pháp và không được tái sử dụng cho mục đích khác ngoài phạm vi đã tuyên bố ban đầu.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
	Team Leader (Trưởng nhóm) hoặc Project Manager (Quản lý dự án / Quản lý chất lượng - QA Lead) phụ trách trực tiếp bộ dữ liệu đó.

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
