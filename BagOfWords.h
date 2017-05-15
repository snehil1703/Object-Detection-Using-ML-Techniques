//
// Created by Snehil Vishwakarma on 4/2/17.
//

class BagOfWords : public Classifier
{
public:

    BagOfWords(const vector<string> &_class_list) : Classifier(_class_list) {}

    virtual void train(const Dataset &filenames)
    {
        vector < vector < SiftDescriptor > > set;
        vector < SiftDescriptor > bagofwords;
        for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
        {
            vector < SiftDescriptor > temp_descript;
            int randrow = rand() % (c_iter->second.size());
            CImg<double> input_image(c_iter->second[randrow].c_str());
            input_image.resize(size, size, 1, 3);
            CImg<double> gray = input_image.get_RGBtoHSI().get_channel(2);
            temp_descript = Sift::compute_sift(gray);
            int randdescript = rand() % (temp_descript.size());
            bagofwords.push_back(temp_descript[randdescript]);
            for(int i=0; i< c_iter->second.size(); ++i)
            {
                CImg<double> temp_input_image(c_iter->second[i].c_str());
                temp_input_image.resize(size,size,1,3);
                CImg<double> temp_gray = temp_input_image.get_RGBtoHSI().get_channel(2);
                vector < SiftDescriptor > tempset(Sift::compute_sift(temp_gray));
                set.push_back(tempset);
            }
        }

        float *_descriptor = new float[128];
        memset(_descriptor,0,128);
        SiftDescriptor zero(0,0,0,0,_descriptor);
        for(int i=0; i<(zero.descriptor.size()); ++i)
            zero.descriptor[i]=0;
        int iterations = 0;
        vector < int > tempval(bagofwords.size(),0);
        vector < vector < int > > finalvalue;

        while(true)
        {
            vector < SiftDescriptor > tempbagofwords(bagofwords.size(),zero);
            vector < int > count(bagofwords.size(),0);
            vector < vector < int > > value(set.size(),tempval);
            for(int i=0; i< set.size(); ++i)
            {
                int classmaxindex = -1,classval=0;
                for(int k=0; k<(set[i].size()); ++k)
                {
                    float min_dist = FLT_MAX;
                    int index = -1;
                    for (int j = 0; j < bagofwords.size(); ++j)
                    {
                        double dist = 0;
                        for (int pos = 0; pos < (bagofwords[j].descriptor.size()); ++pos)
                            dist += pow(set[i][k].descriptor[pos] - bagofwords[j].descriptor[pos], 2);
                        if (min_dist > dist)
                        {
                            min_dist = dist;
                            index = j;
                        }
                    }
                    for (int j = 0; j < tempbagofwords[index].descriptor.size(); ++j)
                        tempbagofwords[index].descriptor[j] += set[i][k].descriptor[j];
                    (count[index])++;
                    (value[i][index])++;
                    if(value[i][index]>classval)
                    {
                        classval=value[i][index];
                        classmaxindex=index;
                    }
                }
            }
            bool change = false;
            for(int i=0; i< (tempbagofwords.size()); ++i)
            {
                for (int j = 0; j < (tempbagofwords[i].descriptor.size()); ++j)
                {
                    tempbagofwords[i].descriptor[j] /= ((float)(count[i]));
                    if ((tempbagofwords[i].descriptor[j] != bagofwords[i].descriptor[j]) && !change)
                        change = true;
                }
            }
            if(change && iterations<size)
                bagofwords = tempbagofwords;
            else
            {
                bagofwords = tempbagofwords;
                finalvalue = value;
                break;
            }
            iterations++;
        }
        cout << "Iterations done: " << iterations << endl;

        // Visual Vocabulary - vector < vector < float > descriptor > bagofwords
        system ("rm -f bow.in");
        system ("touch bow.in");
        for(int i=0; i<(bagofwords.size()); ++i)
        {
            stringstream ss (stringstream::in | stringstream::out);
            ss <<"echo \"";
            for(int j=0; j<(bagofwords[i].descriptor.size()); ++j)
            {
                ss << bagofwords[i].descriptor[j] <<" ";
            }
            ss << "\" >> bow.in";
            string tstring = ss.str();
            system(tstring.c_str());
        }


        // Histogram for training - vector < vector < int > > finalvalue
        //SVM Classifier
        int ct = 0,classct=1;
        system ("rm -f svm_learning.in");
        system ("touch svm_learning.in");
        for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
        {
            for(int i=0; i< c_iter->second.size(); ++i)
            {
                stringstream ss (stringstream::in | stringstream::out);
                ss <<"echo \""<<classct;
                for(int j=0; j<(finalvalue[ct].size()); ++j)
                {
                    if(finalvalue[ct][j] != 0)
                        ss <<" "<<(j+1)<<":"<<finalvalue[ct][j];
                }
                ss << "\" >> svm_learning.in";
                string tstring = ss.str();
                system(tstring.c_str());
                ct++;
            }
            classct++;
        }
        system ("./svm_multiclass_learn -c 10.0 -t 2 -g 0.0625 svm_learning.in svm_learnt.in");
    }

    virtual string classify(const string &filename)
    {
        vector < SiftDescriptor > test_descript = extract_features(filename);
        vector < int > value(class_list.size(),0);
        int classmaxindex = -1,classval = 0;
        for(int i=0; i<(test_descript.size()); ++i)
        {
            float min_dist = FLT_MAX;
            int index = -1;
            for (int c = 0; c < class_list.size(); ++c)
            {
                double dist = 0;
                for (int pos = 0; pos < (models[class_list[c]].size()); ++pos)
                    dist += pow(test_descript[i].descriptor[pos] - models[class_list[c]][pos], 2);
                if (min_dist > dist)
                {
                    min_dist = dist;
                    index = c;
                }
            }
            (value[index])++;
            if(value[index]>classval)
            {
                classval=value[index];
                classmaxindex=index;
            }
        }
        // Histogram for testing - vector < int > value
        size_t s1 = filename.find_first_of('/');
        string ts = filename.substr(s1+1,string::npos);
        ts = ts.substr(0,ts.find_first_of('/'));
        int c;
        for (c = 0; c < class_list.size(); ++c)
        {
            if(ts == class_list[c])
                break;
        }
        system ("rm -f svm_testing.in");
        system ("rm -f svm_tested.in");
        system ("touch svm_testing.in");
        stringstream ss (stringstream::in | stringstream::out);
        ss <<"echo \""<<(c+1);
        for(int j=0; j<(value.size()); ++j)
        {
            if(value[j] != 0)
                ss << " " <<(j+1) << ":" << value[j];
        }
        ss << "\" >> svm_testing.in";
        string tstring = ss.str();
        system(tstring.c_str());

        system ("./svm_multiclass_classify svm_testing.in svm_learnt.in svm_tested.in");

        fstream source;
        source.open("svm_tested.in", ios_base::in);
        string temp2;
        getline(source,temp2);
        stringstream ss2(temp2);
        int svm_class_detected;
        ss2 >> svm_class_detected;
        cout << "svm_tested: " << temp2 << endl;
        cout << "svm_class_detected: " << svm_class_detected-1 << endl;
        source.close();
        return class_list[svm_class_detected-1];
    }

    virtual void load_model()
    {
        fstream source;
        source.open("bow.in", ios_base::in);
        for(int c=0; c < class_list.size(); c++)
        {
            string temp;
            getline(source,temp);
            stringstream ss(temp);
            float t;
            vector < float > temp_model;
            while(ss >> t)
                temp_model.push_back(t);
            models[class_list[c] ] = temp_model;
        }
        source.close();
    }

protected:

    vector < SiftDescriptor > extract_features(const string &filename)
    {
        CImg<double> input_image(filename.c_str());
        input_image.resize(size, size, 1, 3);
        CImg<double> gray = input_image.get_RGBtoHSI().get_channel(2);
        vector < SiftDescriptor > temp_descript = Sift::compute_sift(gray);
        return temp_descript;
    }

    static const int size=300;  // subsampled image resolution
    map < string, vector < float > > models; // trained models
};