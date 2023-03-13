# NOTE
Tensorboard 사용시, 클래스가 다르더라도 label이 달라야 한다 


+ BAD
```python
writer.log_audio(log_noisy,"noisy",step)
writer.log_audio(log_estim,"estim",step)
writer.log_audio(log_clean,"clean",step)

writer.log_spec(log_noisy,"noisy",step)
writer.log_spec(log_estim,"estim",step)
writer.log_spec(log_clean,"clean",step)
```
=> audio에서도 "noisy" 라벨 붙은걸 다 띄우려하고, image에서도 "noisy" 붙은 거를 다 띄우려한다. 

+ GOOD
```python
writer.log_audio(log_noisy,"noisy_audio",step)
writer.log_audio(log_estim,"estim_audio",step)
writer.log_audio(log_clean,"clean_audio",step)

writer.log_spec(log_noisy,"noisy_spec",step)
writer.log_spec(log_estim,"estim_spec",step)
writer.log_spec(log_clean,"clean_spec",step)
```
