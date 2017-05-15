function bow_desc = get_bow(W,k)
bow_desc = zeros(k,1);
for i = 1:k
    bow_desc(i) = sum(W == i);
end
end