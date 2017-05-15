#include "fstream"


#define extract_features_overfeat 0

class Deep : public Classifier
{
public:
  Deep(const std::vector<string> &_class_list) : Classifier(_class_list){}

  virtual void train(const Dataset &filenames)
  {
    // map classes to class number for svm file format
    int num = 1;
    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
    {
      classnum.insert(pair<string,int>(c_iter->first, num) );
      num++;
    }

    // extract features using overfeat and put into input file format for svm
    // set extract_features_overfeat = 1 to extract features using CNN again
    if (extract_features_overfeat)
    {
        system("rm input");
        system("rm tempfeature");
        fstream fw;
        for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
        {
          cout << "Processing " << c_iter->first << endl;
          CImg<double> class_vectors(size*size*3,c_iter->second.size(),1);
          for(int i=0; i<c_iter->second.size(); i++)
          {
            fw.open("input",ios::app);
            fw << classnum[c_iter->first] << " ";
            fw.close();
        	  extract_features(c_iter->second[i].c_str());
          }
        }

        system("mv input train_features");
    }

    system("./svm_light/svm_multiclass_linux64/svm_multiclass_learn -c 3.2 -t 2 -g 0.0625 train_features model_file");

  }
  virtual string classify(const string &filename)
  {
    size_t t1 = filename.find_first_of('/');
    string ts = filename.substr(t1+1, string::npos);
    ts = ts.substr(0,ts.find_first_of('/'));
    int c;
    for(c = 0; c < class_list.size(); c++)
    {
      if(class_list[c] == ts)
        break;
    }
    fstream fw, fr;
    fw.open("input",ios::out);
    fw << c+1 << " ";
    fw.close();
    extract_features(filename.c_str());
    system("./svm_light/svm_multiclass_linux64/svm_multiclass_classify input model_file test_prediction");

    fr.open("test_prediction",ios::in);
    int cls;
    fr >> cls;
    fr.close();
    system("rm test_prediction");
    return class_list[cls];
  }

  virtual void load_model()
  {

  }

protected:
  static const int size=231;
  map<string, int > classnum;
  // extract feature from an image - resample the image and use overfeat to extract features
  CImg<double> extract_features(const string &filename)
  {
    CImg<double> temp = (CImg<double>(filename.c_str())).resize(size,size,1,3);
    CImg<double> i;
    temp.save("deeptemp.jpg");
    system("./overfeat/bin/linux_64/overfeat -L 10 deeptemp.jpg | tail -n +2 > tempfeature");

    fstream fr, fw;
    fr.open("tempfeature",ios::in);
    fw.open("input",ios::app);

    if (fr!=NULL)
    {
      string value;
      int cnt = 1;
      std::string str;
      while (fr >> value) {
        if (value != "0")
        fw << cnt << ":" << value << " ";
        cnt++;
      }
    }
    fw << '\n';
    fr.close();
    fw.close();

    return i;

  }
};
