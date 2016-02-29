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
BOOST = /opt/boost/boost_1_53_0
INTEL = /opt/intel/Compiler/13.1/2.183

# compiler binary
CXX = g++

# for older gcc versions, replace 'c++11' with 'c++0x'
CXXFLAGS = -std=c++11 -O2 -m64 -mtune=native -fopenmp \
           -I. -I$(BOOST)/include \
           -I$(INTEL)/ipp/include \
           -I$(INTEL)/mkl/include

# Please check whether you will need the two lines commented out by '##'.
LDFLAGS := -L$(BOOST)/lib -lboost_program_options -lboost_filesystem \
                          -lboost_system -lboost_random -lboost_iostreams -lm \
           -L$(INTEL)/compiler/lib/intel64 -liomp5 \
           -L$(INTEL)/ipp/lib/intel64 -lippcore -lipps \
           -L$(INTEL)/mkl/lib/intel64 -lmkl_rt \
           -lpthread \
           -Wl,-rpath,$(BOOST)/lib \
           -Wl,-rpath,$(INTEL)/compiler/lib/intel64 \
           -Wl,-rpath,$(INTEL)/ipp/lib/intel64 \
           -Wl,-rpath,$(INTEL)/mkl/lib/intel64
##           -Wl,-rpath,$(INTEL)/lib/intel64 \
##           -L$(INTEL)/lib/intel64 -liomp5 \

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
