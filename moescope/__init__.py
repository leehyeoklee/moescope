"""moescope — MoE 모델을 EP 방식으로 여러 GPU에 나누어 서빙하고 layer별 GPU 간 통신량을 측정하는 도구.

측정한 통신량으로 입력 분야와 응답 토큰을 추정하는 분류기를 학습·평가한다. EP로 서빙한 출력은
한 GPU에서 실행한 출력과 같고, 측정한 통신량은 라우팅으로 계산한 값과 일치한다.

모듈
  config      모델 사양(층·Expert·top-k·revision)
  placement   Expert→rank 선형 배치, 라우팅→통신량 계산값
  traffic     통신량 특징
  ep          실제 all_to_all 디스패치 + 측정 (gloo/NCCL)
  serve_ep    모델을 GPU별 프로세스에 나누어 EP로 서빙하며 통신량 측정
  classifiers 통신량→분야/토큰 분류기
  chat        EP 서빙 입력창과 실시간 화면
  cli         `moescope` 명령
"""
__version__ = "0.1.0"
