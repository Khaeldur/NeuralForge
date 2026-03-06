# NeuralForge CLI — Makefile
CC = xcrun clang
CFLAGS = -O2 -Wall -DACCELERATE_NEW_LAPACK -fobjc-arc -Wno-deprecated-declarations
VENDOR = ../vendor/ANE/training
FRAMEWORKS = -framework Foundation -framework CoreML -framework IOSurface -framework Accelerate
LDFLAGS = $(FRAMEWORKS) -ldl -lobjc

neuralforge: main.m config.h progress.h tokenizer.h \
             $(VENDOR)/stories_config.h $(VENDOR)/stories_io.h \
             $(VENDOR)/stories_mil.h $(VENDOR)/stories_cpu_ops.h \
             $(VENDOR)/ane_rmsnorm_bwd.h $(VENDOR)/ane_classifier.h
	$(CC) $(CFLAGS) -I$(VENDOR) -o $@ main.m $(LDFLAGS)

test_cli: test_cli.m config.h progress.h tokenizer.h $(VENDOR)/stories_config.h
	$(CC) -O0 -g -Wall -DACCELERATE_NEW_LAPACK -fobjc-arc -Wno-deprecated-declarations \
		-I$(VENDOR) -DNEURALFORGE_TEST -o $@ test_cli.m $(LDFLAGS)

test: test_cli
	./test_cli

clean:
	rm -f neuralforge test_cli

.PHONY: clean test
