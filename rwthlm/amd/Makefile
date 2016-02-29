#
# Copyright 2014 RWTH Aachen University. All rights reserved.
# 
# Licensed under the RWTH LM License (the "License");
# you may not use this file except in compliance with the License.
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
SRC = data.cc identity.cc main.cc recurrency.cc softmax.cc tanh.cc \
      vocabulary.cc gradienttest.cc linear.cc output.cc sigmoid.cc \
      tablelookup.cc trainer.cc net.cc htklatticerescorer.cc lstm.cc
OBJ = $(SRC:%.cc=%.o)
DEPENDFILE = .depend
BOOST =   /opt/boost/boost_1_53_0
ACML  =   /opt/acml-5.3.1
AMDLIBM = /opt/amdlibm-3-0-2

# compiler binary
CXX = g++

# compiler and linker options
CXXFLAGS = -std=c++0x -O2 -m64 -mtune=native -fopenmp \
           -I. -I$(BOOST)/include \
           -I$(ACML)/gfortran64_mp/include \
           -I$(ACML)/CBLAS/include \
           -I$(AMDLIBM)/include
LDFLAGS := -L$(BOOST)/lib -lboost_program_options -lboost_filesystem \
                          -lboost_system -lboost_random -lboost_iostreams -lm \
           -L$(ACML)/gfortran64_mp/lib -lacml_mp \
           -L$(ACML)/CBLAS/lib -lcblas_acml \
           -L$(AMDLIBM)/lib/dynamic -lamdlibm \
           -lpthread \
           -Wl,-rpath,$(BOOST)/lib \
           -Wl,-rpath,$(ACML)/gfortran64_mp/lib \
           -Wl,-rpath,$(ACML)/CBLAS/lib \
           -Wl,-rpath,$(AMDLIBM)/lib/dynamic \

# default rule
all: dep rwthlm

-include $(DEPENDFILE) # "-" means do not stop if file does not exist

rwthlm: $(OBJ)

.PHONY: clean dep

clean:
	rm -f *.o rwthlm

dep: $(SRC)
	$(CXX) $(CXXFLAGS) -MM $(SRC) > $(DEPENDFILE)

# implicit rules (-n show commands, -r no implicit rules)
rwthlm: $(OBJ)
	$(CXX) $(CXXFLAGS) -o rwthlm $(OBJ) $(LDFLAGS)
$(OBJ): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ -c $(@:%.o=%.cc)
