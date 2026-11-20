"""moescope — MoE expert-parallel 추론의 GPU 간 통신량 누출 연구 툴.

핵심 주장: EP all_to_all로 실제 보낸 목적지 GPU별 토큰 수(= 통신량)는 라우팅으로
계산한 값과 **정확히 일치**한다. 따라서 통신량만 관찰해도 라우팅이 드러나고, 학습된
분류기로 입력 분야·출력 토큰을 복원할 수 있다.

모듈
  config      모델 사양(층·Expert·top-k·revision)
  placement   Expert→rank 선형 배치, 라우팅→통신량 계산값
  traffic     통신량 특징
  ep          실제 all_to_all 디스패치 + 측정 (gloo/NCCL)
  collect     실제 모델에서 라우팅 추출
  serve_ep    모델을 GPU별 프로세스에 나누어 EP로 서빙하며 통신량 측정
  classifiers 통신량→분야/토큰 분류기
  cli         `moescope` 명령
"""
__version__ = "0.1.0"
