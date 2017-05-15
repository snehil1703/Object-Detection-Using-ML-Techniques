function bow_desc = get_bow2(X,C)
num_desc = size(X,2);
bow_desc = zeros(size(C,2),1);
for i = 1:num_desc
    a = C - repmat(X(:,i),1,size(C,2));
    dist = diag(a'*a);
    [~,idx] = min(dist);
    bow_desc(idx) = bow_desc(idx) + 1;
end
end