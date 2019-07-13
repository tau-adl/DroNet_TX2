SHELL=/bin/bash -o pipefail
TARGET?=$(shell uname -m)
LIBDIR?=lib
LOCALLIB?="../../lib"
VERBOSE?=1
ifeq ($(VERBOSE), 1)
AT=
else
AT=@
endif
TRIPLE=x86_64-linux
ifeq ($(TARGET), aarch64)
TRIPLE=aarch64-linux
endif
ifeq ($(TARGET), qnx)
TRIPLE=qnx-linux
endif
ifeq ($(TARGET), android32)
TRIPLE=arm-linux-androideabi
endif
ifeq ($(TARGET), android64)
TRIPLE=aarch64-linux-android
endif
export TARGET
export VERBOSE
export LIBDIR
export TRIPLE
export LOCALLIB
solibs=trtinference



.PHONY: all clean help test test_debug test_release
all:
	$(AT)$(foreach solib,$(solibs), $(MAKE) -C $(solib) &&) :

clean:
	$(AT)$(foreach app,$(apps), $(MAKE) clean -C $(app) &&) :
	$(AT)$(foreach solib,$(solibs), $(MAKE) clean -C $(solib) &&) :

test: test_debug test_release

test_debug:
	$(AT)$(foreach app,$(apps), $(MAKE) test_debug -C $(app) &&) :

test_release:
	$(AT)$(foreach app,$(apps), $(MAKE) test_release -C $(app) &&) :

solibs:
	$(AT)$(foreach solib,$(solibs), $(MAKE) solib -C $(solib) &&) :

help:
	$(AT)echo "Applications building help menu."
	$(AT)echo "Apps:"
	$(AT)$(foreach app,$(apps), echo "\t$(app)" &&) :
	$(AT)echo "\nCommands:"
	$(AT)echo "\tall - build all samples."
	$(AT)echo "\tsolibs - build all shared libraries."
	$(AT)echo "\tclean - clean all samples."
	$(AT)echo "\ttest[_debug|_release] - test all/release/debug samples."
	$(AT)echo "\nVariables:"
	$(AT)echo "\tTARGET - Specify the target to build for."
	$(AT)echo "\tVERBOSE - Specify verbose output."
	$(AT)echo "\tCUDA_INSTALL_DIR - Directory where cuda installs to."
