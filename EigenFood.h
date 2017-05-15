//
// Created by Snehil Vishwakarma on 4/1/17.
//

class EigenFood : public Classifier
{
public:
    EigenFood(const vector<string> &_class_list) : Classifier(_class_list) {}

    // Nearest neighbor training. All this does is read in all the images, resize
    // them to a common size, convert to greyscale, and dump them as vectors to a file
    virtual void train(const Dataset &filenames)
    {

        //CImg<double> input_image(inputFile.c_str());
        //CImg<double> gray = input_image.get_RGBtoHSI().get_channel(2);
        //vector < SiftDescriptor > descriptors = Sift::compute_sift(gray);

        for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
        {
            cout << "Processing " << c_iter->first << endl;
            CImg<double> class_vectors(size*size*3, c_iter->second.size(), 1);
            for(int i=0; i<c_iter->second.size(); i++)
                class_vectors = class_vectors.draw_image(0, i, 0, 0, extract_features(c_iter->second[i].c_str()));

            CImg<double> mean_vectors(size*size*3, 1, 1);
            for(int i=0; i<size*size*3; i++)
            {
                mean_vectors(i,0)=0;
                for(int j=0; j<c_iter->second.size(); j++)
                    mean_vectors(i,0)+=class_vectors(i,j);
                mean_vectors(i,0)/=(c_iter->second.size());
            }

            CImg<double> temp(size*size*3, c_iter->second.size(), 1);
            for(int i=0; i<c_iter->second.size(); i++)
                for(int j=0; j<size*size*3; j++)
                    temp(j,i) = class_vectors(j,i)-mean_vectors(j,0);
            CImg<double> tempT(temp);
            tempT.transpose();
            CImg<double> A=tempT*temp;
            CImg<double> val, vec;
            A.symmetric_eigen(val,vec);
            int k = 49;
            CImg<double> vec_topk( k, vec._height, 1);
            cout <<"\n HI1 \n";
            for(int i=0; i<k; i++)
                for(int j=0; j<vec._height; j++)
                    vec_topk(j,i)=vec(j,i);
            cout <<"\n HI2 \n";

            //cout << endl << "A: " << A._width <<" " << A._height << " " << A._depth << " " << A._spectrum <<endl;
            /*int c;
            for(c=0; c<val._height; c++)
                if(val(c,0)<1)
                    break;
            cout <<c<<endl;*/
            //cout << endl << "val: " << val._width <<" " << val._height << " " << val._depth << " " << val._spectrum <<endl;
            /*for(int i=0; i<val._width; i++)
            {
                cout <<endl;
                for(int j=0; j<val._height; j++)
                    cout<<val(j,i)<<" ";
            }*/
            //cout << endl << "vec: " << vec._width <<" " << vec._height << " " << vec._depth << " " << vec._spectrum <<endl;
            /*for(int i=0; i<vec._width; i++)
            {
                cout <<endl;
                for(int j=0; j<vec._height; j++)
                    cout<<vec(j,i)<<" ";
            }*/
            //for(int i=0; i<vec._width; i++)
            //{
            //    vec = vec.draw_image(0, i, 0, 0, extract_features(c_iter->second[i].c_str()));
            //    vec_topk = vec_topk.draw_image(0, i, 0, 0, extract_features(c_iter->second[i].c_str()));
            //}
            vec.normalize(0,255).save_png(("ef_model_vec." + c_iter->first + ".png").c_str());
            cout <<"\n HI3 \n";

            vec_topk.normalize(0,255).save_png(("ef_model_vec_topk." + c_iter->first + ".png").c_str());
            cout <<"\n HI4 \n";
        }
    }

    virtual string classify(const string &filename)
    {
        CImg<double> test_image = extract_features(filename);

        // figure nearest neighbor
        pair<string, double> best("", 10e100);
        double this_cost;
        for(int c=0; c<class_list.size(); c++)
            for(int row=0; row<models[ class_list[c] ].height(); row++)
                if((this_cost = (test_image - models[ class_list[c] ].get_row(row)).magnitude()) < best.second)
                    best = make_pair(class_list[c], this_cost);

        return best.first;
    }

    virtual void load_model()
    {
        for(int c=0; c < class_list.size(); c++)
            models[class_list[c] ] = (CImg<double>(("ef_model." + class_list[c] + ".png").c_str()));
    }
protected:
    // extract features from an image, which in this case just involves resampling and
    // rearranging into a vector of pixel data.
    CImg<double> extract_features(const string &filename)
    {
        return (CImg<double>(filename.c_str())).resize(size,size,1,3).unroll('x');
    }

    static const int size=20;  // subsampled image resolution
    map<string, CImg<double> > models; // trained models
};
