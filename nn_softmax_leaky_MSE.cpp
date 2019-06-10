// Copyright 2019 <Jony>
#include <time.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <map>

void generateSample(int predict_num, std::vector<std::vector<float> > &sample) {
  for (int i=0; i < 100; ++i) {
    std::vector<float> var(8);
    int a = 180, b = 255;
    int c = a + rand() % (b-a+1);
    int d = a + rand() % (b-a+1);

    a = 0;
    b = 30;
    float c1 = static_cast<float>(a + rand() % (b-a+1));
    float d1 = static_cast<float>(a + rand() % (b-a+1));
    float c2 = static_cast<float>(a + rand() % (b-a+1));
    float d2 = static_cast<float>(a + rand() % (b-a+1));
    float c3 = static_cast<float>(a + rand() % (b-a+1));
    float d3 = static_cast<float>(a + rand() % (b-a+1));

    if (predict_num == 1) {
      var[0] = c1/256.0;
      var[1] = c2/256.0;
      var[2] = c/256.0;
      var[3] = c3/256.0;
      var[4] = d1/256.0;
      var[5] = d2/256.0;
      var[6] = d/256.0;
      var[7] = d3/256.0;
    } else if (predict_num == 2) {
      var[0] = c/256.0;
      var[1] = c1/256.0;
      var[2] = c2/256.0;
      var[3] = c3/256.0;
      var[4] = d/256.0;
      var[5] = d1/256.0;
      var[6] = d2/256.0;
      var[7] = d3/256.0;
    } else if (predict_num == 3) {
      var[0] = c1/256.0;
      var[1] = c/256.0;
      var[2] = c2/256.0;
      var[3] = c3/256.0;
      var[4] = d1/256.0;
      var[5] = d/256.0;
      var[6] = d2/256.0;
      var[7] = d3/256.0;
    } else if (predict_num == 4) {
      var[0] = c1/256.0;
      var[1] = c2/256.0;
      var[2] = c3/256.0;
      var[3] = c/256.0;
      var[4] = d1/256.0;
      var[5] = d2/256.0;
      var[6] = d3/256.0;
      var[7] = d/256.0;
    }
    sample.push_back(var);
  }
}

class activationFunc {
 public:
  float a, z, t;
  activationFunc() {
    z = 0.0;
    a = 0.0;
    t = 0.0;
  }
  float leaky(std::vector<float> &w, std::vector<float> &x, float &b) {
    int wsize = w.size();
    for (int i=0; i < wsize; ++i) {
      z += w[i] * x[i];
    }
    z = z + b;
    if (z > 0) a = z;
    else
      a = 0.001 * z;
    t = exp(z);
    return a;
  }
  float leakyDerivative(float a) {
    if (a > 0) return 1;
    else
      return 0.001;
  }
};

class nnNode {
 public:
  float A, L, oldL, dz, bias, db, da, t;
  std::vector<float> dw, weights;
  explicit nnNode(int n) {
    weights.resize(n);
    dw.resize(n);
    bias = 0.0;
    A = 0.0;
    da = 0.0;
    dz = 0.0;
    db = 0.0;
    L = 0.0;
    oldL = 0.0;
    t = 0.0;
    int weightssize = weights.size();
    for (int i=0; i < weightssize; ++i) {
      weights[i] = (float)(rand() % 100) / 10000.0;
      dw[i] = 0.0;
    }
    bias = (float)(rand() % 100) / 10000.0;
  }
  float FP(std::vector<float> &X, int Y, bool cal_loss) {
    activationFunc af;
    A = af.leaky(weights, X, bias);
    if (cal_loss) {
      t = af.t;
      L += 0.5 * pow((A - Y), 2);
      da = A - Y;
    }
    return A;
  }
  void printWeights() {
    std::cout << "weights is ";
    for (auto i : weights) std::cout << i << ", ";
    std::cout << std::endl;
  }
};

class netlayer {
 public:
  std::vector<std::vector<nnNode> > layer;
  std::vector<std::vector<float> > output;
  std::vector<int> layernum;
  int inputDim;
  float learningRate;
  explicit netlayer(float lr, std::vector<std::vector<nnNode> > ly,
                    std::vector<std::vector<float> > out, std::vector<int> lnum, int inputdim) {
    this->learningRate = lr;
    this->layer = ly;
    this->output = out;
    this->layernum = lnum;
    this->inputDim = inputdim;
    std::vector<float> temp2(inputdim, 0.0);
    output.push_back(temp2);
    int lnumsize = lnum.size();
    for (int i=0; i < lnumsize; ++i) {
      std::vector<nnNode> temp;
      for (int j=0; j < lnum[i]; ++j) {
        if (i == 0) {
          nnNode nnn1(inputdim);
          temp.push_back(nnn1);
        } else {
          nnNode nnn1(lnum[i-1]);
          temp.push_back(nnn1);
        }
      }
      layer.push_back(temp);
      std::vector<float> temp3(lnum[i], 0.0);
      output.push_back(temp3);
    }
  }
  void FP(std::vector<float> onesam, int layerid, int Y) {
    // store output of each node in this layer
    std::vector<float> temp;
    bool cal_loss = false;
    float sumt = 0.0;
    int layernodesize = layer[layerid].size();
    for (int nodenum=0; nodenum < layernodesize; ++nodenum) {
      int temp_Y;
      int layersize = layer.size();
      if (layerid == layersize-1) {
        if (Y == nodenum) {
          temp_Y = 1;
        } else {
          temp_Y = 0;
        }
        cal_loss = true;
      } else {
        cal_loss = false;
      }
      layer[layerid][nodenum].FP(onesam, temp_Y, cal_loss);
      temp.push_back(layer[layerid][nodenum].A);
      sumt += layer[layerid][nodenum].t;
    }
    for (int nodenum=0; nodenum < layernodesize; ++nodenum) {
      layer[layerid][nodenum].t /= sumt;
    }
    int tempsize = temp.size();
    for (int i=0; i < tempsize; ++i)
      output[layerid+1][i] = temp[i];
  }
  void updateWsBs(int currentlayerid, int m) {
    int layercursize = layer[currentlayerid].size();
    for (int node_i=0; node_i < layercursize; ++node_i) {
      int layersize = layer.size();
      if (currentlayerid == layersize-1) {
        layer[currentlayerid][node_i].L /= m;
        layer[currentlayerid][node_i].oldL = layer[currentlayerid][node_i].L;
        layer[currentlayerid][node_i].L = layer[currentlayerid][node_i].oldL*0.9 + layer[currentlayerid][node_i].L*0.1;
        std::cout << "loss is " << layer[currentlayerid][node_i].L << std::endl;
      }
      int layerdwsize = layer[currentlayerid][node_i].dw.size();
      for (int i=0; i < layerdwsize; ++i) {
        layer[currentlayerid][node_i].dw[i] /= m;
        layer[currentlayerid][node_i].weights[i] -= learningRate * layer[currentlayerid][node_i].dw[i];
      }
      layer[currentlayerid][node_i].db /= m;
      layer[currentlayerid][node_i].bias -= learningRate * layer[currentlayerid][node_i].dz;
    }
  }
  void BP(int clid) {
    float w_dz = 0.0, w_temp = 0.0;
    activationFunc af;
    int layercursize = layer[clid].size();
    for (int node_i=0; node_i < layercursize; ++node_i) {
      int layersize = layer.size();
      if (clid == layersize-1) {
        w_dz = layer[clid][node_i].da;
      } else {
        int layernextsize = layer[clid+1].size();
        for (int wt=0; wt < layernextsize; ++wt) {
          w_dz = layer[clid+1][wt].weights[node_i];
          w_dz *= layer[clid+1][wt].dz;
          w_temp += w_dz;
        }
        w_dz = w_temp;
      }
      layer[clid][node_i].dz = w_dz * af.leakyDerivative(layer[clid][node_i].A);
      layer[clid][node_i].db += layer[clid][node_i].dz;
      int layerdwsize = layer[clid][node_i].dw.size();
      for (int j=0; j < layerdwsize; ++j) {
        layer[clid][node_i].dw[j] +=
            layer[clid][node_i].dz * static_cast<float>(output[clid][j]);
      }
    }
  }
};

int main() {
  srand((unsigned)time(NULL));
  std::vector<std::vector<float> > Xall;
  generateSample(1, Xall);  // 0
  generateSample(2, Xall);  // 1
  generateSample(3, Xall);  // 2
  generateSample(4, Xall);  // 3
  std::vector<int> Y(400);
  for (int i=0; i < 4; ++i) {
    for (int j=0; j < 100; ++j) {
      Y[i*100 + j] = i;
    }
  }
  int Xallsize = Xall.size();
  int maxiter(3000);
  int batch = 20;
  int subbatch = Xallsize / batch;
  std::vector<std::vector<nnNode> > ly;
  std::vector<std::vector<float> > out;
  std::vector<int> lnum{25, 10, 4};
  int inputdim = 8;
  netlayer netlayer(0.01, ly, out, lnum, inputdim);

  int netlayersize = netlayer.layer.size();
  int lnumsize = lnum.size();
  for (int maxiter_i = 0; maxiter_i < maxiter; ++maxiter_i) {
    for (int ba=0; ba < batch; ++ba) {
      for (int m=0; m < subbatch; ++m) {
        std::vector<float> var = Xall[ba*subbatch + m];
        // FP
        for (int layerid=0; layerid < netlayersize; ++layerid) {
          if (layerid == 0) {
            int varsize = var.size();
            for (int var_i=0; var_i < varsize; ++var_i)
              netlayer.output[0][var_i] = var[var_i];
          }
          netlayer.FP(netlayer.output[layerid], layerid, Y[ba*subbatch+m]);
        }
        // BP
        for (int i=lnumsize-1; i >= 0; --i) {
          netlayer.BP(i);
        }
      }
      for (int i=lnumsize-1; i >= 0; --i) {
        netlayer.updateWsBs(i, Y.size());
      }
    }
  }

  // test
  std::vector<std::vector<float> > Xall2;
  generateSample(1, Xall2);  // 0
  generateSample(2, Xall2);  // 1
  generateSample(3, Xall2);  // 2
  generateSample(4, Xall2);  // 3
  int err = 0;
  for (int m=0; m < Xallsize; ++m) {
    std::vector<float> var = Xall2[m];
    // FP
    for (int layerid=0; layerid < netlayersize; ++layerid) {
      if (layerid == 0) {
        netlayer.FP(var, layerid, Y[m]);
      } else {
        netlayer.FP(netlayer.output[layerid], layerid, Y[m]);
      }
    }
    std::map<float, int> softmax;
    int netlayerbacksize = netlayer.layer.back().size();
    for (int i=0; i < netlayerbacksize; ++i) {
      softmax.insert(std::pair<float, int>(netlayer.layer.back()[i].t, i));
    }

    for (auto si = softmax.rbegin(); si != softmax.rend(); ++si) {
      std::cout << "sample " << Y[m] << " predict as";
      std::cout << " class " << si->second << " probability is " << si->first << std::endl;
    }
    std::cout << std::endl;
    if (Y[m] != softmax.rbegin()->second) ++err;
  }
  std::cout << "accuracy is " << (1.0-err/300.0) << std::endl;
  return 0;
}
