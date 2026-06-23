# BootCamp 교육 - "Embedded Linux 기반 On-Ddevice AI" 프로젝트

- On-Device AI 관련 프로젝트를 교육 기간 내 진행해야 한다. 

## Context

- raw plan (초기 brain storming): 
무신사(의류 쇼핑몰)에서 옷 이미지 크롤링 잔뜩해서 데이터 베이스 만들어놓은 뒤, 웹캠이 찍은 사람 전신에서 상체 / 하체 / 신발 등등 인식해서 입은 옷을 인식해서 이미지 임베딩모델 유사도로 줄 세워서 님한테 어울리는 옷 실시간으로 추천해주기. 목표는 LG Styler에서 사용하기 위함. 

---

## Goal

- 데모 수준의 프로젝트 완성

---

## 사용 가능 장비 설명


- 라즈베리파이 5 (최대 4개 사용 가능)
| 항목    | 확인 결과                        |
| ----- | ------------------------------  |
| 보드    | Raspberry Pi 5 Model B Rev 1.0 |
| CPU   | ARM Cortex-A76, 4코어            |
| 아키텍처  | aarch64, 64-bit ARM           |
| RAM   | 8GB 모델                         |
| 저장장치  | 32GB microSD 카드              |
| OS    | Debian GNU/Linux 12 bookworm    |
| 커널    | Linux 6.6.62+rpt-rpi-2712      |
| 현재 온도 | 47.7°C                        |
| 스로틀링  | 없음                           |

- 각 보드에 카메라 연동되어 있으며, 사용해도 되고 안해도 된다. 
- 현재, 각 보드에 `/Users/SeungHyunLee/.ssh/config`에서 아래의 ip로 ssh 접속하여 개발 중이다:
```
Host LGBoard
    HostName 10.56.130.185
    User willtek
    ForwardX11 yes
    ForwardX11Trusted yes
```

다른 3개의 보드에 대한 IP는 각각 다음과 같다:
- 10.56.130.178
- 10.56.130.182
- (미확인)